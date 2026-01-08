# video classification model as outlined in paper
# usage: python videoclassificationmodel.py, change hyperparameters as needed with argparse/in code
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import h5py
import time
import os
import torchvision.models as models
import wandb
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# dataset class to load in input/output
class WindowedDataset(Dataset):
    def __init__(self, images_all, aux_all, targets_all, window_size, valid_indices=None):
        self.images_all = images_all
        self.aux_all = aux_all
        self.targets_all = targets_all
        self.window_size = window_size
        if valid_indices is None:
            self.valid_indices = np.arange(window_size - 1, len(images_all))
        else:
            self.valid_indices = np.array([i for i in valid_indices if i >= window_size - 1])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        window_imgs = self.images_all[i - self.window_size + 1 : i + 1]
        aux_window = self.aux_all[i - self.window_size + 1 : i + 1]
        target = self.targets_all[i]  # already 0/1
        return (
            torch.tensor(window_imgs, dtype=torch.float32),
            torch.tensor(aux_window, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)  # scalar
        )

# model
class ResNetBackbone(nn.Module):
    def __init__(self, in_channels=1, emb_dim=256,
                 pretrained_path="/scratch/gpfs/th5879/model/resnet18_imagenet.pth"):
        super().__init__()

        self.resnet = models.resnet18(weights=None)

        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained weights not found at: {pretrained_path}")

        state_dict = torch.load(pretrained_path, map_location='cpu')

        # conversion to in_channels
        old_conv = self.resnet.conv1
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )

            # adapt checkpoint weights BEFORE load
            if in_channels == 1 and "conv1.weight" in state_dict:
                w = state_dict["conv1.weight"]  # (64,3,7,7)
                state_dict["conv1.weight"] = w.mean(dim=1, keepdim=True)  # (64,1,7,7)

        # now load safely
        missing, unexpected = self.resnet.load_state_dict(state_dict, strict=False)

        # remove FC layer for embedding output
        self.resnet.fc = nn.Identity()

        # small projection head
        self.proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.proj(self.resnet(x))

class SEPSeqModel(nn.Module):
    def __init__(
        self,
        image_channels=1,
        image_emb=256,
        aux_dim=2,
        aux_emb=32,
        n_heads=4,
        n_attn_blocks=3,
        hidden_head=256,
        dropout=0.1,
        window_size=8
    ):
        super().__init__()
        # image backbone
        self.image_backbone = ResNetBackbone(in_channels=image_channels, emb_dim=image_emb)

        # auxiliary features
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, aux_emb),
            nn.ReLU(),
            nn.Linear(aux_emb, aux_emb),
            nn.ReLU()
        )

        # concatenate embeddings
        joint_dim = image_emb + aux_emb
        self.frame_proj = nn.Linear(joint_dim, hidden_head)

        # learnable temporal embeddings
        self.time_emb = nn.Parameter(torch.zeros(window_size, hidden_head))
        nn.init.trunc_normal_(self.time_emb, std=0.02)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_head,
            nhead=n_heads,
            dim_feedforward=hidden_head*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_attn_blocks
        )

        # pooling and final head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_head, hidden_head),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, 1)
        )

    def forward(self, imgs, aux):
        """
        imgs: (B, T, C, H, W)
        aux:  (B, T, aux_dim)
        returns: (B, 1)
        """
        B, T, C, H, W = imgs.shape

        # per-frame image embedding
        img_emb = self.image_backbone(imgs.view(B*T, C, H, W)).view(B, T, -1)

        # auxiliary features embedding
        aux_emb = self.aux_mlp(aux.view(B*T, -1)).view(B, T, -1)

        # concatenate image + aux embeddings
        frame = self.frame_proj(torch.cat([img_emb, aux_emb], dim=-1))  # (B, T, hidden_head)

        # add temporal embeddings
        x = frame + self.time_emb.unsqueeze(0)  # (B, T, hidden_head)

        # pass through transformer
        x = self.transformer(x)  # (B, T, hidden_head)

        # pool across time
        x = x.transpose(1, 2)           # (B, hidden_head, T)
        x = self.pool(x).squeeze(-1)   # (B, hidden_head)

        out = self.head(x)
        return out


def preload_matched_images_into_memory(df, cache_file=None, max_file=None):

    if cache_file is None:
        cache_file = "/scratch/gpfs/th5879/data_collection/matched_images_171.pkl"
    if max_file is None:
        max_file = "/scratch/gpfs/th5879/data_collection/matched_images_max_171.pkl"

    # load cached results if available
    if os.path.exists(cache_file) and os.path.exists(max_file):
        with open(cache_file, 'rb') as f:
            images_all = pickle.load(f)
        with open(max_file, 'rb') as f:
            max_val = pickle.load(f)
        print(f"[DEBUG] Loaded cached 171 images ({len(images_all)})")
        return images_all, max_val

    # open HDF5 file
    with h5py.File("/scratch/gpfs/th5879/data_collection/aia171_images_3hr_cadence.h5", "r") as f171:
        times_171 = pd.to_datetime(np.array(f171["T_OBS"], dtype=str), errors='coerce')
        print(f"[DEBUG] 171 timestamps: {times_171.min()} → {times_171.max()}")

        images_list = []
        max_171 = 0.0

        for t in tqdm(df["SDO_time"], desc="Preloading 171 images"):
            # nearest 171 timestamp
            idx171 = np.argmin(np.abs((times_171 - t).total_seconds()))
            img171 = np.nan_to_num(f171["images"][idx171][...], nan=0.0)
            img171 = np.clip(img171, 0, None)
            max_171 = max(max_171, img171.max())
            images_list.append(img171)

        # log-normalize
        images_all = np.array([
            np.log1p(img) / np.log1p(max_171) for img in images_list
        ], dtype=np.float16)

        # add channel dim: (N, 1, H, W)
        images_all = images_all[:, np.newaxis, :, :]

    # cache results
    with open(cache_file, 'wb') as f:
        pickle.dump(images_all, f)
    with open(max_file, 'wb') as f:
        pickle.dump(max_171, f)

    print(f"[DEBUG] Loaded {len(images_all)} 171 images, max_val={max_171:.3f}")
    return images_all, max_171

# metrics
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TN, FP, FN, TP = cm.ravel()
    acc = (TP + TN) / max((TP + TN + FP + FN), 1)
    precision = TP / max((TP + FP), 1)
    recall = TP / max((TP + FN), 1)
    far = FP / max((FP + TN), 1)
    tss = recall - far
    denom = (TP + FN)*(FN + TN) + (TP + FP)*(FP + TN)
    hss = (2*(TP*TN - FP*FN))/denom if denom != 0 else 0
    return cm, acc, precision, recall, far, tss, hss

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig

# main
def main():
    CSV_PATH = "/scratch/gpfs/th5879/data_collection/final_psp_df_3hr_cadence.csv"
    df = pd.read_csv(CSV_PATH)
    df['SDO_time'] = pd.to_datetime(df['SDO_time'])
    df = df[df["photo_captures_footprint"] != 0].reset_index(drop=True)
    df = df.dropna(subset=["epilo_jlinlin_offset"]).reset_index(drop=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--window_size", type=int, default=8)
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    window_size = args.window_size

    name = (
        f"TIMECLASSep{epochs}_bs{batch_size}_lr{learning_rate}_winsize{window_size}"
    )
    os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project="psp-sep-prediction",
        name=name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "adam",
            "learning_rate": learning_rate,
            "loss": "BCEWithLogitsLoss",
        }
    )

    # binary targets
    threshold = 1e-1
    y = (df["epilo_jlinlin_offset"].values > threshold).astype(np.float32)
    aux_features = np.stack([
        df["psp_footpoint_stonyhurst_lon"].values.astype(np.float32)/180.0,
        df["psp_ephem_features_HCI_R"].values.astype(np.float32)
    ], axis=1)

    # images
    images_all, max_val = preload_matched_images_into_memory(df)

    all_indices = np.arange(len(df))
    blocks = [all_indices[i:i+80] for i in range(0, len(all_indices), 80)]
    rng = np.random.default_rng(seed=1717)
    rng.shuffle(blocks)
    train_cutoff = int(0.8*len(blocks))
    train_idx = np.concatenate([block for block in blocks[:train_cutoff]])
    val_idx   = np.concatenate([block for block in blocks[train_cutoff:]])

    # set up data loaders to load in samples to model
    train_ds = WindowedDataset(images_all, aux_features, y, window_size, train_idx)
    val_ds   = WindowedDataset(images_all, aux_features, y, window_size, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SEPSeqModel(window_size=window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    num_pos = y.sum()
    num_neg = len(y) - num_pos
    pos_weight = torch.tensor(num_neg / max(num_pos, 1), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_epoch = 0
    MODEL_OUT = f"/scratch/gpfs/th5879/model/models/sep_{name}.pt"

    # train model for specified epochs
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for imgs, aux, yb in train_loader:
            imgs, aux, yb = imgs.to(device), aux.to(device), yb.view(-1,1).to(device)
            optimizer.zero_grad()
            preds = model(imgs, aux)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, aux, yb in val_loader:
                imgs, aux, yb = imgs.to(device), aux.to(device), yb.view(-1,1).to(device)
                preds = model(imgs, aux)
                val_losses.append(criterion(preds, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        })

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_OUT)
        elif epoch - best_epoch > 30:
            print(f"No improvement after 30 epochs, stopping early.")
            break


    # load best model
    model.load_state_dict(torch.load(MODEL_OUT))
    model.eval()

    # containers
    y_train_pred, y_train_true = [], []
    y_val_pred, y_val_true = [], []

    with torch.no_grad():
        # train evaluation
        for imgs, aux, yb in train_loader:
            preds = model(imgs.to(device), aux.to(device))
            y_train_pred.append(preds.cpu().numpy())
            y_train_true.append(yb.numpy())
        
        # val evaluation
        for imgs, aux, yb in val_loader:
            preds = model(imgs.to(device), aux.to(device))
            y_val_pred.append(preds.cpu().numpy())
            y_val_true.append(yb.numpy())

    with torch.no_grad():
        y_train_pred_raw = []
        for imgs, aux, yb in train_loader:
            preds = model(imgs.to(device), aux.to(device))
            y_train_pred_raw.append(preds.cpu().numpy())

        # stack into a single array
        y_train_pred_raw = np.concatenate([yp.reshape(-1) for yp in y_train_pred_raw])

        # check min/max/mean values
        print("Train predictions (raw):")
        print("min:", y_train_pred_raw.min())
        print("max:", y_train_pred_raw.max())
        print("mean:", y_train_pred_raw.mean())
        print("first 20:", y_train_pred_raw[:20])

    # stack into arrays
    y_train_true = np.concatenate([yb.reshape(-1) for yb in y_train_true]).astype(int)
    y_val_true   = np.concatenate([yb.reshape(-1) for yb in y_val_true]).astype(int)

    # convert logits to 0/1 predictions
    y_train_pred = (np.concatenate([yp.reshape(-1) for yp in y_train_pred]) > 0.0).astype(int)
    y_val_pred   = (np.concatenate([yp.reshape(-1) for yp in y_val_pred]) > 0.0).astype(int)

    # compute metrics
    cm_train, acc_train, prec_train, rec_train, far_train, tss_train, hss_train = compute_metrics(y_train_true, y_train_pred)
    cm_val, acc_val, prec_val, rec_val, far_val, tss_val, hss_val = compute_metrics(y_val_true, y_val_pred)

    # print metrics
    print("\nClassification statistics:\n")
    print("---- Train ----")
    print(f"Accuracy:           {acc_train:.4f}")
    print(f"Precision:          {prec_train:.4f}")
    print(f"Recall:             {rec_train:.4f}")
    print(f"False Alarm Rate:   {far_train:.4f}")
    print(f"TSS:                {tss_train:.4f}")
    print(f"HSS:                {hss_train:.4f}")
    print(f"Confusion Matrix:\n{cm_train}\n")

    print("---- Validation ----")
    print(f"Accuracy:           {acc_val:.4f}")
    print(f"Precision:          {prec_val:.4f}")
    print(f"Recall:             {rec_val:.4f}")
    print(f"False Alarm Rate:   {far_val:.4f}")
    print(f"TSS:                {tss_val:.4f}")
    print(f"HSS:                {hss_val:.4f}")
    print(f"Confusion Matrix:\n{cm_val}\n")

    # plot confusion matrices
    fig_train_cm = plot_confusion_matrix(cm_train, "Train Confusion Matrix")
    fig_val_cm = plot_confusion_matrix(cm_val, "Validation Confusion Matrix")
    plt.show()

    wandb.log({
        "train_confusion_matrix": wandb.Image(fig_train_cm),
        "val_confusion_matrix": wandb.Image(fig_val_cm),
    })
    wandb.log({
        "train_accuracy": acc_train,
        "train_precision": prec_train,
        "train_recall": rec_train,
        "train_false_alarm_rate": far_train,
        "train_tss": tss_train,
        "train_hss": hss_train,
        "val_accuracy": acc_val,
        "val_precision": prec_val,
        "val_recall": rec_val,
        "val_false_alarm_rate": far_val,
        "val_tss": tss_val,
        "val_hss": hss_val
    })


if __name__ == "__main__":
    main()