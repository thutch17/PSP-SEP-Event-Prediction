# video regression model as outlined in paper
# usage: python videoregressionmodel.py, change hyperparameters as needed with argparse/in code
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

        # slice the window: (time, C, H, W)
        window_imgs = self.images_all[i - self.window_size + 1 : i + 1]

        aux_window = self.aux_all[i - self.window_size + 1 : i + 1]  # (T, aux_dim)        # (aux_dim,)
        target = self.targets_all[i] # scalar or (1,)

        return (
            torch.tensor(window_imgs, dtype=torch.float32),  # (T, C, H, W)
            torch.tensor(aux_window, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)        # scalar
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

# full model definition
class SEPSeqModel(nn.Module):
    def __init__(
        self,
        image_channels=1,
        image_emb=256,
        aux_dim=2,
        aux_emb=32,
        time_emb_dim=64,
        n_heads=4,
        n_attn_blocks=3,
        hidden_head=256,
        dropout=0.1,
        use_binary=False,
        window_size=8
    ):
        super().__init__()
        self.image_backbone = ResNetBackbone(in_channels=image_channels, emb_dim=image_emb)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, aux_emb),
            nn.ReLU(),
            nn.Linear(aux_emb, aux_emb),
            nn.ReLU()
        )
        # concatenate embeddings
        joint_dim = image_emb + aux_emb
        self.frame_proj = nn.Linear(joint_dim, hidden_head)
        self.time_emb = nn.Parameter(torch.zeros(window_size, hidden_head))
        nn.init.trunc_normal_(self.time_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_head,
            nhead=n_heads,
            dim_feedforward=hidden_head*4,
            dropout=dropout,
            batch_first=True  # important so input shape is (B, T, E)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_attn_blocks
        )

        # pooling across time and final head
        self.pool = nn.AdaptiveAvgPool1d(1)
        # final linear layers for regression or classification
        self.head = nn.Sequential(
            nn.Linear(hidden_head, hidden_head),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, 1)
        )
        self.use_binary = use_binary
        if use_binary:
            self.classifier = nn.Sigmoid()

    def forward(self, imgs, aux):
        """
        imgs: (B, T, C, H, W)
        aux:  (B, T, aux_dim)
        returns: (B, 1)
        """
        B, T, C, H, W = imgs.shape
        # per-frame image embedding
        imgs_reshaped = imgs.view(B * T, C, H, W)

        img_emb = self.image_backbone(imgs_reshaped)# (B*T, image_emb)
        img_emb = img_emb.view(B, T, -1)# (B, T, image_emb)

        # aux embedding per timestep
        aux_emb = self.aux_mlp(aux.view(B*T, -1)).view(B, T, -1)# (B,T,aux_emb)

        # concat image + aux
        frame = torch.cat([img_emb, aux_emb], dim=-1)  # (B,T, joint_dim)
        frame = self.frame_proj(frame)# (B,T, hidden_head)

        # add temporal embeddings
        # time_emb: (T, E) -> broadcast to (B, T, E)
        x = frame + self.time_emb.unsqueeze(0)  # (B, T, hidden_head)

        # pass through TransformerEncoder
        x = self.transformer(x)  # (B, T, hidden_head)

        # pool across time: convert to (B, hidden_head)
        x = x.transpose(1,2)            # (B, hidden_head, T)
        x = self.pool(x).squeeze(-1)    # (B, hidden_head)

        out = self.head(x)# (B,1)
        if self.use_binary:
            return self.classifier(out)
        return out

def preload_matched_images_into_memory(df, cache_file=None, max_file=None):
    """
    Load AIA171 images corresponding to the PSP dataframe, normalize, and cache.
    Returns:
        images_all: np.array of shape (N, 1, H, W), log-normalized to [0,1]
        max_val: scalar maximum value used for normalization
    """
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

def invert_log_scaling(y_scaled, y_mean, y_std):
    # y_log = y_scaled * y_std + y_mean
    return np.expm1(y_scaled)

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    # accuracy, precision, recall
    acc = (TP + TN) / max((TP + TN + FP + FN), 1)

    precision = TP / max((TP + FP), 1)
    recall    = TP / max((TP + FN), 1)

    # false alarm rate
    far = FP / max((FP + TN), 1)

    # true skill score
    tss = recall - far

    # Heidke Skill Score
    denom = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    hss = (2 * (TP * TN - FP * FN)) / denom if denom != 0 else 0

    return cm, acc, precision, recall, far, tss, hss

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig


def plot_pred_vs_actual(y_true, y_pred, dates_mpl, name="epilo", title_prefix="Train"):
    mask = (y_true[:, 0] > 0) & (y_pred[:, 0] > 0)
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(
        y_true[mask, 0],
        y_pred[mask, 0],
        c=dates_mpl[mask],
        cmap='viridis',
        alpha=0.7
    )
    lims = [y_true[mask, 0].min(), y_true[mask, 0].max()]
    ax.plot(lims, lims, 'r--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f"Actual {name}")
    ax.set_ylabel(f"Predicted {name}")
    ax.set_title(f"{title_prefix}: Predicted vs Actual {name} (colored by date)")
    ax.grid(True, which="both", ls="--")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("date (matplotlib date number)")
    return fig

def main():
    CSV_PATH = "/scratch/gpfs/th5879/data_collection/final_psp_df_3hr_cadence.csv"
    df = pd.read_csv(CSV_PATH)
    df['SDO_time'] = pd.to_datetime(df['SDO_time'])

    df = df[df["photo_captures_footprint"] != 0].reset_index(drop=True)
    df = df.dropna(subset=["epilo_jlinlin"]).reset_index(drop=True)

    # parse hyperparameters
    parser = argparse.ArgumentParser(
        description="Train SEP prediction model using PSP and SDO/AIA data."
    )

    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout probability")
    parser.add_argument("--train_block_size", type=int, default=80, help="Training block size")
    parser.add_argument("--train_fraction", type=float, default=1, help="Fraction of each training block to use (0 < p <= 1)")
    parser.add_argument("--window_size", type=int, default=8, help="Window size")

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout_rate = args.dropout
    train_block_size = args.train_block_size
    train_fraction = args.train_fraction
    window_size = args.window_size

    print(f"""
    === training configuration ===
    epochs:            {epochs}
    batch_size:        {batch_size}
    learning_rate:     {learning_rate}
    dropout_rate:      {dropout_rate}
    train_block_size:  {train_block_size}
    train_fraction:    {train_fraction}
    window_size:       {window_size}
    ==============================
    """)

    name = (
        f"TRANSep{epochs}_bs{batch_size}_lr{learning_rate}"
        f"_drop{dropout_rate}_trainbatch{train_block_size}"
    )
    MODEL_OUT = f"/scratch/gpfs/th5879/model/models/sep_prediction_{name}.pt"
    os.environ["WANDB_MODE"] = "offline"

    # set device to use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="psp-sep-prediction",
        name=name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "adam",
            "learning_rate": learning_rate,
            "loss": "mse",
            "dropout_rate": dropout_rate,
        }
    )

    y_full = df[["epilo_jlinlin_offset"]].values.astype(np.float32)
    y_log = np.log1p(y_full)
    y_mean = np.mean(y_log, axis=0)
    y_std = np.std(y_log, axis=0)
    # y = (y_log - y_mean) / y_std
    y = y_log

    aux_lon = df["psp_footpoint_stonyhurst_lon"].values.astype(np.float32) / 180.0
    aux_r = df["psp_ephem_features_HCI_R"].values.astype(np.float32)
    aux_features = np.stack([aux_lon, aux_r], axis=1)  # shape (N, 2)

    images_all, max_val = preload_matched_images_into_memory(df)
    # valid indices after filtering
    all_indices = np.arange(0, len(df)) 
    T = window_size

    # create contiguous blocks of size train_block_size
    blocks = [all_indices[i:i+train_block_size] for i in range(0, len(all_indices), train_block_size)]

    # shuffle blocks
    rng = np.random.default_rng(seed=1717)
    rng.shuffle(blocks)

    # split 80/20 into train/val
    train_cutoff = int(0.8 * len(blocks))
    train_blocks = blocks[:train_cutoff]
    val_blocks   = blocks[train_cutoff:]

    train_blocks = [b for b in train_blocks]
    val_blocks   = [b for b in val_blocks]

    train_idx = []
    for block in train_blocks:
        n = len(block)
        step = max(1, int(1 / args.train_fraction))
        selected = block[::step]
        train_idx.append(selected)

    # flatten blocks back to indices
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_blocks)
    timestamps = df["SDO_time"].to_numpy()

    # build train and validation targets
    y_train = y[train_idx]
    y_val = y[val_idx]

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    # build windowed dataset/loaders to keep images in memory only once
    train_ds = WindowedDataset(images_all, aux_features, y, window_size, valid_indices=train_idx)
    val_ds   = WindowedDataset(images_all, aux_features, y, window_size, valid_indices=val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print("built all loaders and such")

    # model definition
    model = SEPSeqModel(
        image_channels=1, # just 171 for now
        image_emb=256,
        aux_dim=2,
        aux_emb=32,
        n_heads=4,
        n_attn_blocks=3,
        hidden_head=256,
        dropout=dropout_rate,
        use_binary=False,
        window_size=window_size
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # training loop with early stopping
    best_val_loss = float("inf")
    best_epoch = 0
    MODEL_OUT = f"/scratch/gpfs/th5879/model/models/sep_time{name}.pt"
    print("starting training")

    for epoch in range(epochs):
        # train
        model.train()
        train_losses = []

        for imgs, aux, yb in train_loader:
            imgs = imgs.to(device)# (B, T, C, H, W)
            
            yb  = yb.view(-1, 1).to(device) # ensures shape (B, 1) to match targets
            aux = aux.to(device)
            # normalize times relative to first frame in window
            B, T, C, H, W = imgs.shape

            optimizer.zero_grad()
            preds = model(imgs, aux)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, aux, yb in val_loader:
                imgs = imgs.to(device)
                aux = aux.to(device)
                yb  = yb.squeeze(-1).view(-1, 1).to(device)  # match (B,1) shape

                B, T, C, H, W = imgs.shape

                preds = model(imgs, aux)
                val_losses.append(criterion(preds, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0]
        })

        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # save best model & early stopping
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


    # evaluate final predictions based on best model (no qualitative plots)
    with torch.no_grad():
        y_train_pred, y_train_true_phys = [], []
        for Xb, auxb, yb in tqdm(train_loader, desc="Evaluating train set"):
            # create time embeddings for SEPSeqModel
            B, T, C, H, W = Xb.shape

            preds = model(Xb.to(device), auxb.to(device))
            y_train_pred.append(preds.cpu().numpy())
            y_train_true_phys.append(yb.cpu().numpy())

        y_train_pred = np.vstack(y_train_pred)
        y_train_true_phys = invert_log_scaling(np.vstack(y_train_true_phys), y_mean, y_std)

        y_val_pred, y_val_true_phys = [], []
        for Xb, auxb, yb in tqdm(val_loader, desc="Evaluating val set"):
            B, T, C, H, W = Xb.shape

            preds = model(Xb.to(device), auxb.to(device))
            y_val_pred.append(preds.cpu().numpy())
            y_val_true_phys.append(yb.cpu().numpy())

        y_val_pred = np.vstack(y_val_pred)
        y_val_true_phys = invert_log_scaling(np.vstack(y_val_true_phys), y_mean, y_std)

    # inverse-transform predictions
    y_train_pred_phys = invert_log_scaling(y_train_pred, y_mean, y_std)
    y_val_pred_phys = invert_log_scaling(y_val_pred, y_mean, y_std)

    # classification thresholding
    threshold = 1e-1
    y_train_pred_bin = (y_train_pred_phys[:, 0] > threshold).astype(int)
    y_train_true_bin = (y_train_true_phys[:, 0] > threshold).astype(int)
    y_val_pred_bin = (y_val_pred_phys[:, 0] > threshold).astype(int)
    y_val_true_bin = (y_val_true_phys[:, 0] > threshold).astype(int)

    # compute classification metrics
    cm_train, acc_train, prec_train, rec_train, far_train, tss_train, hss_train = \
        compute_metrics(y_train_true_bin, y_train_pred_bin)


    cm_val, acc_val, prec_val, rec_val, far_val, tss_val, hss_val = \
        compute_metrics(y_val_true_bin, y_val_pred_bin)

    fig_train_cm = plot_confusion_matrix(cm_train, title="Train Confusion Matrix")
    fig_val_cm   = plot_confusion_matrix(cm_val, title="Validation Confusion Matrix")

    wandb.log({
        "train_confusion_matrix": wandb.Image(fig_train_cm),
        "val_confusion_matrix": wandb.Image(fig_val_cm)
    })

    # log metrics to wandb
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

    train_dates_mpl = mdates.date2num(timestamps[train_ds.valid_indices])
    val_dates_mpl   = mdates.date2num(timestamps[val_ds.valid_indices])

    # log train plot
    fig_train = plot_pred_vs_actual(y_train_true_phys, y_train_pred_phys, train_dates_mpl, title_prefix="Train")
    wandb.log({"train_pred_vs_actual_epilo": wandb.Image(fig_train)})
    plt.close(fig_train)

    # log validation plot
    fig_val = plot_pred_vs_actual(y_val_true_phys, y_val_pred_phys, val_dates_mpl, title_prefix="Val")
    wandb.log({"val_pred_vs_actual_epilo": wandb.Image(fig_val)})
    plt.close(fig_val)

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

    wandb.finish()
    print(f"Training complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()