# modified version of snapshot model with some optimizations
# usage: python snapshotmodel2.py
# modify hyperparameters with argparse or in code

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

import torch
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random

import wandb

H5_PATH = "/scratch/gpfs/th5879/data_collection/aia171_images_3hr_cadence.h5"
CSV_PATH = "/scratch/gpfs/th5879/data_collection/final_psp_df_3hr_cadence.csv"

# simple dataset for 2D images and scalar features
class SimpleDataset(Dataset):
    def __init__(self, images_all, aux_all, targets_all, valid_indices=None):
        self.images_all = images_all
        self.aux_all = aux_all
        self.targets_all = targets_all

        if valid_indices is None:
            self.valid_indices = np.arange(len(images_all))
        else:
            self.valid_indices = np.array(valid_indices)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        img = self.images_all[i]            # shape: (channels, H, W)
        aux = self.aux_all[i]               # shape: (num_aux_features,)
        target = self.targets_all[i]        # shape: (num_targets,)
        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(aux, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )

def main():
    # parse hyperparameters
    parser = argparse.ArgumentParser(
        description="Train SEP prediction model using PSP and SDO/AIA data."
    )

    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--num_dense_nodes", type=int, default=64, help="Number of dense layer neurons")
    parser.add_argument("--num_conv", type=int, default=5, help="Number of convolution layers")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout probability")
    parser.add_argument("--train_block_size", type=int, default=80, help="Training block size")
    parser.add_argument("--train_fraction", type=float, default=1, help="Fraction of each training block to use (0 < p <= 1)")
    parser.add_argument("--num_dense_layers", type=int, default=2, help="Number of dense layers")

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_dense_nodes = args.num_dense_nodes
    num_conv = args.num_conv
    dropout_rate = args.dropout
    train_block_size = args.train_block_size
    train_fraction = args.train_fraction
    num_dense_layers = args.num_dense_layers
    print(f"""
    === training configuration ===
    epochs:            {epochs}
    batch_size:        {batch_size}
    learning_rate:     {learning_rate}
    num_dense_nodes:   {num_dense_nodes}
    num_conv:          {num_conv}
    dropout_rate:      {dropout_rate}
    train_block_size:  {train_block_size}
    train_fraction:    {train_fraction}
    num_dense_layers:  {num_dense_layers}
    ==============================
    """)

    name = (
        f"SSep{epochs}_bs{batch_size}_lr{learning_rate}_dnodes{num_dense_nodes}_dlays{num_dense_layers}_conv{num_conv}"
        f"_drop{dropout_rate}_trainbatch{train_block_size}"
    )
    MODEL_OUT = f"/scratch/gpfs/th5879/model/models/sep_prediction_{name}.pt"
    os.environ["WANDB_MODE"] = "offline"

    # set device to use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # setup w&b output
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

    print("loading PSP dataframe...")
    df = pd.read_csv(CSV_PATH)
    df['SDO_time'] = pd.to_datetime(df['SDO_time'])

    # filter out where image does not capture PSP footprint & nans
    df = df[df["photo_captures_footprint"] != 0].reset_index(drop=True)
    df = df.dropna(subset=["epilo_jlinlin_offset"]).reset_index(drop=True)

    print("loading images from HDF5 file")
    with h5py.File(H5_PATH, "r") as f:
        images_dset = f["images"]
        times = np.array(f["T_OBS"], dtype=str)

    print("converting timestamps...")
    times = pd.to_datetime(times, errors='coerce')
    mask = ~times.isna()
    valid_indices = np.where(mask)[0]
    times = times[mask]

    # match up PSP rows with their corresponding images
    print("matching PSP times to image times...")
    df = df.sort_values("SDO_time").reset_index(drop=True)
    matched_idx = []
    for t in tqdm(df["SDO_time"], desc="Matching times"):
        deltas = np.abs((times - t).total_seconds())
        idx = np.argmin(deltas)
        matched_idx.append(valid_indices[idx])

    df["img_index"] = matched_idx
    print("Filtered dataframe shape:", df.shape)


    # for speed, load & normalize images from h5 into RAM
    print("Preloading all matched images into memory...")
    images_all, norm_factors = preload_matched_images_into_memory(df)
    print(f"Loaded {len(images_all)} images")
    print(f"Per-channel max values: {norm_factors}")

    aux_features = df["psp_footpoint_stonyhurst_lon"].values.astype(np.float32) / 180.0
    aux_features = aux_features.reshape(-1, 1)   # shape (N, 1)

    # log normalize the prediction targets
    y_full = df[["epilo_jlinlin_offset"]].values.astype(np.float32)
    y_log = np.log1p(y_full)
    y_mean = np.mean(y_log, axis=0)
    y_std = np.std(y_log, axis=0)
    # y = (y_log - y_mean) / y_std
    y = y_log


    # valid indices after filtering
    all_indices = np.arange(0, len(df)) 

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
        step = max(1, int(1 / args.train_fraction))  # spacing between indices
        selected = block[::step]
        train_idx.append(selected)

    # flatten blocks back to indices
    train_idx = np.concatenate(train_idx)
    val_idx   = np.concatenate(val_blocks)

    # build train and validation targets
    y_train = y[train_idx]
    y_val = y[val_idx]

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    

    timestamps = df["SDO_time"].to_numpy()

    images_train = images_all[train_idx]
    aux_train = aux_features[train_idx]
    y_train = y[train_idx]

    images_val = images_all[val_idx]
    aux_val = aux_features[val_idx]
    y_val = y[val_idx]

    train_ds = SimpleDataset(images_all, aux_features, y, valid_indices=train_idx)
    val_ds   = SimpleDataset(images_all, aux_features, y, valid_indices=val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # model definition
    class SEPModel(nn.Module):
        def __init__(self, num_conv, num_dense_nodes, dropout_rate, in_channels, num_dense_layers):
            super().__init__()
            filters = [16, 32, 64, 128, 256][:num_conv]

            conv_layers = []
            for f in filters:
                conv_layers += [
                    nn.Conv2d(in_channels, f, kernel_size=3, padding=1),
                    nn.BatchNorm2d(f),
                    nn.ReLU(),
                    nn.MaxPool2d(2)  # spatial pooling
                ]
                in_channels = f
            self.conv = nn.Sequential(*conv_layers)
            self.global_pool = nn.AdaptiveAvgPool2d((1,1))
            # self.global_pool = nn.AdaptiveMaxPool2d((1,1))
            self.post_pool_dropout = nn.Dropout(dropout_rate)
            
            # dense layers
            layers = []
            input_dim = filters[-1] + 1
            # first dense layer with BatchNorm
            layers.append(nn.Linear(input_dim, num_dense_nodes))
            layers.append(nn.BatchNorm1d(num_dense_nodes))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            for _ in range(num_dense_layers - 1):
                layers.append(nn.Linear(num_dense_nodes, num_dense_nodes))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(num_dense_nodes, 1))
            self.fc = nn.Sequential(*layers)

        def forward(self, img, aux):
            # img shape: (batch, channels, H, W)
            x = self.conv(img)
            x = self.global_pool(x).view(x.size(0), -1)
            x = self.post_pool_dropout(x)
            x = torch.cat([x, aux], dim=-1)
            return self.fc(x)

    # set up model with command line arguments
    model = SEPModel(num_conv, num_dense_nodes, dropout_rate, 2, num_dense_layers).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    criterion = nn.MSELoss()
    # added learning rate decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=8,
        min_lr=1e-7
    )

    from torchinfo import summary

    # get one sample from training dataset
    sample_img, sample_aux, sample_y = train_ds[0]

    # inspect model summary
    summary(model, input_data=[
        # add batch dimension
        sample_img.unsqueeze(0).to(device),
        sample_aux.unsqueeze(0).to(device)
    ])

    # training loop
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        train_losses = []

        for Xb, auxb, yb in train_loader:
            Xb, auxb, yb = Xb.to(device), auxb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb, auxb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, auxb, yb in val_loader:
                Xb, auxb, yb = Xb.to(device), auxb.to(device), yb.to(device)
                preds = model(Xb, auxb)
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

        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.3e}  val_loss={val_loss:.3e}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_OUT)
            best_epoch = epoch
        elif epoch - best_epoch > 50:
            print("no improvement after 50 epochs, early stopping")
            break


    model.load_state_dict(torch.load(MODEL_OUT))
    model.eval()

    # evaluate final predictions based on best model
    with torch.no_grad():
        y_train_pred, y_train_true_phys = [], []
        for Xb, auxb, yb in tqdm(train_loader, desc="Evaluating train set"):
            preds = model(Xb.to(device), auxb.to(device))
            y_train_pred.append(preds.cpu().numpy())
            y_train_true_phys.append(yb.cpu().numpy())
        y_train_pred = np.vstack(y_train_pred)
        y_train_true_phys = invert_log_scaling(np.vstack(y_train_true_phys), y_mean, y_std)

        y_val_pred, y_val_true_phys = [], []
        for Xb, auxb, yb in tqdm(val_loader, desc="Evaluating val set"):
            preds = model(Xb.to(device), auxb.to(device))
            y_val_pred.append(preds.cpu().numpy())
            y_val_true_phys.append(yb.cpu().numpy())
        y_val_pred = np.vstack(y_val_pred)
        y_val_true_phys = invert_log_scaling(np.vstack(y_val_true_phys), y_mean, y_std)

    # inverse-transform predictions too
    y_train_pred_phys = invert_log_scaling(y_train_pred, y_mean, y_std)
    y_val_pred_phys = invert_log_scaling(y_val_pred, y_mean, y_std)

    # use the valid_indices from the datasets
    train_dates_mpl = mdates.date2num(timestamps[train_ds.valid_indices])
    val_dates_mpl   = mdates.date2num(timestamps[val_ds.valid_indices])

    for i, name in enumerate(["epilo"]):
        # plot actual/predicted for training set
        mask_train = (y_train_true_phys[:, i] > 0) & (y_train_pred_phys[:, i] > 0)
        fig_train, ax_train = plt.subplots(figsize=(6,6))
        sc = ax_train.scatter(
            y_train_true_phys[mask_train, i],
            y_train_pred_phys[mask_train, i],
            c=train_dates_mpl[mask_train],
            cmap='viridis',
            alpha=0.7
        )
        lims = [y_train_true_phys[mask_train, i].min(), y_train_true_phys[mask_train, i].max()]
        ax_train.plot(lims, lims, 'r--')
        ax_train.set_xscale('log')
        ax_train.set_yscale('log')
        ax_train.set_xlabel(f"Actual {name}")
        ax_train.set_ylabel(f"Predicted {name}")
        ax_train.set_title(f"Train: Predicted vs Actual {name} (colored by date)")
        ax_train.grid(True, which="both", ls="--")
        cbar = fig_train.colorbar(sc, ax=ax_train)
        cbar.set_label('date (matplotlib date number)')
        wandb.log({f"train_pred_vs_actual_{name}": wandb.Image(fig_train)})
        plt.close(fig_train)

        # plot actual/predicted for validation set
        mask_val   = (y_val_true_phys[:, i] > 0) & (y_val_pred_phys[:, i] > 0)
        fig_val, ax_val = plt.subplots(figsize=(6,6))
        sc = ax_val.scatter(
            y_val_true_phys[mask_val, i],
            y_val_pred_phys[mask_val, i],
            c=val_dates_mpl[mask_val],
            cmap='viridis',
            alpha=0.7
        )
        lims = [y_val_true_phys[mask_val, i].min(), y_val_true_phys[mask_val, i].max()]
        ax_val.plot(lims, lims, 'r--')
        ax_val.set_xscale('log')
        ax_val.set_yscale('log')
        ax_val.set_xlabel(f"Actual {name}")
        ax_val.set_ylabel(f"Predicted {name}")
        ax_val.set_title(f"Val: Predicted vs Actual {name} (colored by date)")
        ax_val.grid(True, which="both", ls="--")
        cbar = fig_val.colorbar(sc, ax=ax_val)
        cbar.set_label('date (matplotlib date number)')
        wandb.log({f"val_pred_vs_actual_{name}": wandb.Image(fig_val)})
        plt.close(fig_val)

    test_dates = [
        # maxima
        "2023-07-25",
        # minima
        "2020-09-02",
        # two random ones
        "2021-08-27",
        "2024-08-27"
    ]
    for j, date_str in enumerate(test_dates):
        date_start = pd.Timestamp(date_str)
        date_end = date_start + pd.Timedelta(days=1)

        mask_date = (df["SDO_time"] >= date_start) & (df["SDO_time"] < date_end)
        subset_df = df[mask_date].sort_values("SDO_time")

        if len(subset_df) == 0:
            print(f"No data found for {date_str}")
            continue

        subset_idx = subset_df.index.values
        df_subset = df.loc[subset_idx].reset_index(drop=True)
        images_subset = images_all[subset_idx]
        aux_subset = aux_features[subset_idx]
        y_subset = y[subset_idx]

        # --- Use your unified window builder ---
        timestamps_subset = df_subset["SDO_time"].to_numpy()
        subset_indices = np.arange(len(df_subset))
        test_samples = generate_samples(
            subset_indices, images_subset, aux_subset, y_subset, timestamps_subset
        )

        if len(test_samples) == 0:
            print(f"Not enough samples for {date_str}")
            continue

        y_true_phys, y_pred_phys, times_sub = [], [], []

        with torch.no_grad():
            for (window_imgs, aux, y_true, t) in test_samples:
                Xb = torch.tensor(window_imgs, dtype=torch.float32).unsqueeze(0).to(device)
                auxb = torch.tensor(aux, dtype=torch.float32).unsqueeze(0).to(device)
                pred = model(Xb, auxb).cpu().numpy()
                y_pred_phys.append(pred)
                y_true_phys.append(y_true)
                times_sub.append(t)

        y_true_phys = invert_log_scaling(np.array(y_true_phys), y_mean, y_std)
        y_pred_phys = invert_log_scaling(np.vstack(y_pred_phys), y_mean, y_std)
        times_sub = pd.to_datetime(times_sub)

        # --- Plotting (same as before) ---
        n = len(subset_idx)
        fig = plt.figure(figsize=(2.5 * n, 6))
        gs = gridspec.GridSpec(2, n, height_ratios=[3, 2])

        fig.suptitle(f"SEP Prediction on {date_start.date()} (3h cadence)", fontsize=14)

        for i in range(n):
            ax_img = fig.add_subplot(gs[0, i])
            img = images_all[subset_idx[i]][0]  # shape: (H, W)
            img = np.expm1(img)
            im = ax_img.imshow(img, cmap="inferno", origin="lower")
            ax_img.set_title(pd.to_datetime(subset_df.iloc[i]["SDO_time"]).strftime("%H:%M"), fontsize=8)
            ax_img.axis("off")

        ax_plot = fig.add_subplot(gs[1, :])
        ax_plot.plot(times_sub, y_true_phys[:, 0], "o-", label="Actual epilo", color="C0")
        ax_plot.plot(times_sub, y_pred_phys[:, 0], "s--", label="Predicted epilo", color="C1")
        ax_plot.set_yscale("log")
        ax_plot.set_xlabel("Time (UT)")
        ax_plot.set_ylabel("epilo flux")
        ax_plot.grid(True, which="both", ls="--", alpha=0.4)
        ax_plot.legend()
        ax_plot.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax_plot.set_xlim(times_sub.min(), times_sub.max())

        plt.tight_layout(rect=[0, 0, 0.9, 0.93])
        wandb.log({f"qualitative_{date_str}": wandb.Image(fig)}, commit=True)
        plt.close(fig)
    
    # classification
    threshold = 1e-1

    y_train_pred_bin = (y_train_pred_phys[:, 0] > threshold).astype(int)
    y_train_true_bin = (y_train_true_phys[:, 0] > threshold).astype(int)

    y_val_pred_bin = (y_val_pred_phys[:, 0] > threshold).astype(int)
    y_val_true_bin = (y_val_true_phys[:, 0] > threshold).astype(int)

    # === TRAIN METRICS ===
    cm_train, acc_train, prec_train, rec_train, far_train, tss_train, hss_train = \
    compute_metrics(y_train_true_bin, y_train_pred_bin)

    fig_cm_train = plot_confusion_matrix(cm_train, title="Train Confusion Matrix")
    wandb.log({
        "train_confusion_matrix": wandb.Image(fig_cm_train),
        "train_accuracy": acc_train,
        "train_precision": prec_train,
        "train_recall": rec_train,
        "train_false_alarm_rate": far_train,
        "train_tss": tss_train,
        "train_hss": hss_train
    })
    plt.close(fig_cm_train)


    # === VAL METRICS ===
    cm_val, acc_val, prec_val, rec_val, far_val, tss_val, hss_val = \
        compute_metrics(y_val_true_bin, y_val_pred_bin)

    fig_cm_val = plot_confusion_matrix(cm_val, title="Validation Confusion Matrix")
    wandb.log({
        "val_confusion_matrix": wandb.Image(fig_cm_val),
        "val_accuracy": acc_val,
        "val_precision": prec_val,
        "val_recall": rec_val,
        "val_false_alarm_rate": far_val,
        "val_tss": tss_val,
        "val_hss": hss_val
    })
    plt.close(fig_cm_val)


    # === PRINT ALL RESULTS AT END ===
    print("\nclassification statistics:\n")

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

    print("==================================================\n")

    wandb.finish()


def preload_matched_images_into_memory(df, cache_file=None, max_file=None):
    if cache_file is None:
        cache_file = "/scratch/gpfs/th5879/data_collection/matched_images_aligned.pkl"
    if max_file is None:
        max_file = "/scratch/gpfs/th5879/data_collection/matched_images_max_aligned.pkl"

    # load cached results
    if os.path.exists(cache_file) and os.path.exists(max_file):
        with open(cache_file, 'rb') as f:
            images_all = pickle.load(f)
        with open(max_file, 'rb') as f:
            max_vals = pickle.load(f)
        print(f"[DEBUG] Loaded cached images ({len(images_all)})")
        return images_all, max_vals

    # open HDF5 files
    with h5py.File("/scratch/gpfs/th5879/data_collection/aia171_images_3hr_cadence.h5", "r") as f171, \
         h5py.File("/scratch/gpfs/th5879/data_collection/aia304_images_3hr_cadence.h5", "r") as f304:

        # read times
        times_171 = pd.to_datetime(np.array(f171["T_OBS"], dtype=str), errors='coerce')
        times_304 = pd.to_datetime(np.array(f304["T_OBS"], dtype=str), errors='coerce')

        print(f"[DEBUG] 171 range: {times_171.min()} → {times_171.max()}")
        print(f"[DEBUG] 304 range: {times_304.min()} → {times_304.max()}")

        images_list = []
        max_171, max_304 = 0.0, 0.0

        for i, t in enumerate(tqdm(df["SDO_time"], desc="Preloading images")):

            # nearest 171 timestamp
            idx171 = np.argmin(np.abs((times_171 - t).total_seconds()))
            t171 = times_171[idx171]

            # nearest 304 timestamp to that 171 timestamp
            idx304 = np.argmin(np.abs((times_304 - t171).total_seconds()))
            t304 = times_304[idx304]

            # load images
            img171 = np.nan_to_num(f171["images"][idx171][...], nan=0.0)
            img304 = np.nan_to_num(f304["images"][idx304][...], nan=0.0)

            img171 = np.clip(img171, 0, None)
            img304 = np.clip(img304, 0, None)

            max_171 = max(max_171, img171.max())
            max_304 = max(max_304, img304.max())

            stacked = np.stack([img171, img304], axis=0)
            images_list.append(stacked)

        # log-normalize per channel
        images_all = np.array([
            np.stack([
                np.log1p(img[0]) / np.log1p(max_171),
                np.log1p(img[1]) / np.log1p(max_304)
            ], axis=0)
            for img in images_list
        ], dtype=np.float16)

    # save cache
    with open(cache_file, 'wb') as f:
        pickle.dump(images_all, f)
    with open(max_file, 'wb') as f:
        pickle.dump((max_171, max_304), f)

    print(f"[DEBUG] Loaded {len(images_all)} images, max_vals=({max_171}, {max_304})")
    return images_all, (max_171, max_304)

# helper to generate samples
def generate_samples(indices, images_all, aux_features, y, timestamps=None):
    """
    Return simple samples: one image per index.
    """
    samples = []
    for i in indices:
        img = images_all[i]        # (channels, H, W)
        aux = aux_features[i]      # (num_aux,)
        target = y[i]              # (num_targets,)
        t = timestamps[i] if timestamps is not None else None
        samples.append((img, aux, target, t))
    return samples


# undo normalization
def invert_log_scaling(y_scaled, y_mean, y_std):
    # y_log = y_scaled * y_std + y_mean
    return np.expm1(y_scaled)

def samples_to_tensors(samples):
    X = np.stack([s[0] for s in samples]).astype(np.float32)      # windowed images
    aux = np.stack([s[1] for s in samples]).astype(np.float32)    # aux features
    y = np.stack([s[2] for s in samples]).astype(np.float32)      # targets
    return torch.tensor(X), torch.tensor(aux), torch.tensor(y)

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


if __name__ == '__main__':
    main()