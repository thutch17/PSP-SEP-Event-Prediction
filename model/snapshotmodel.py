# usage: python snapshotmodel.py
# pass in hyperparameters in argparse or change in code

import os
import sys
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import wandb

import argparse

parser = argparse.ArgumentParser(
    description="Train SEP prediction model using PSP and SDO/AIA data."
)

parser.add_argument("--epochs", type=int, default=500,
                    help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Mini-batch size")
parser.add_argument("--learning_rate", type=float, default=0.0003,
                    help="Learning rate")
parser.add_argument("--num_dense_nodes", type=int, default=64,
                    help="Number of dense layer neurons")
parser.add_argument("--num_conv", type=int, default=5,
                    help="Number of convolution layers")
parser.add_argument("--dropout", type=float, default=0.25,
                    help="Dropout probability")
parser.add_argument("--train_block_size", type=int, default=80,
                    help="Training block size")
parser.add_argument("--train_fraction", type=float, default=1,
                    help="Fraction of each training block to use (0 < p <= 1)")

args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
num_dense = args.num_dense_nodes
num_conv = args.num_conv
dropout_rate = args.dropout
train_block_size = args.train_block_size
train_fraction = args.train_fraction

print(f"""
=== training configuration ===
epochs:            {epochs}
batch_size:        {batch_size}
learning_rate:     {learning_rate}
num_dense_nodes:   {num_dense}
num_conv:          {num_conv}
dropout_rate:      {dropout_rate}
train_block_size:  {train_block_size}
train_fraction:    {train_fraction}
==============================
""")

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    
    accuracy = (TP + TN) / cm.sum()
    false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0

    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
    tss = tpr + tnr - 1

    total = TP + TN + FP + FN
    expected_accuracy = ((TP + FP)*(TP + FN) + (FN + TN)*(FP + TN)) / total**2
    hss = (accuracy - expected_accuracy) / (1 - expected_accuracy) if (1 - expected_accuracy) > 0 else 0

    return cm, accuracy, false_alarm_rate, tss, hss

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig

name = f"ep{epochs}_bs{batch_size}_lr{learning_rate}_dense{num_dense}_conv{num_conv}_drop{dropout_rate}_trainbatch{train_block_size}_epilo"

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
        "architecture": f"{num_conv}conv+pos+dense{num_dense}+dropout{dropout_rate}",
        "loss": "mse",
        "dropout_rate": dropout_rate
    }
)

# data loading and preprocessing
H5_PATH = "/scratch/gpfs/th5879/data_collection/aia171_images_3hr_cadence.h5"
CSV_PATH = "/scratch/gpfs/th5879/data_collection/final_psp_df_3hr_cadence.csv"
MODEL_OUT = f"/scratch/gpfs/th5879/model/models/sep_prediction_{name}.pt"

print("loading PSP dataframe...")
df = pd.read_csv(CSV_PATH)
df['SDO_time'] = pd.to_datetime(df['SDO_time'])

# filter out where image does not capture PSP footprint
print(f"Length before filtering captured_footprint==0: {len(df)}")
df = df[df["photo_captures_footprint"] != 0].reset_index(drop=True)
print(f"Length after filtering captured_footprint==0: {len(df)}")

print(f"Length before filtering NaNs in targets: {len(df)}")
df = df.dropna(subset=["epilo_jlinlin_offset"]).reset_index(drop=True)
print(f"Length after filtering NaNs in targets: {len(df)}")

print("loading images from HDF5 file")
with h5py.File(H5_PATH, "r") as f:
    images_dset = f["images"]
    times = np.array(f["T_OBS"], dtype=str)

print("converting timestamps...")
times = pd.to_datetime(times, errors='coerce')
mask = ~times.isna()
valid_indices = np.where(mask)[0]
times = times[mask]

print("matching PSP times to image times...")
df = df.sort_values("SDO_time").reset_index(drop=True)
matched_idx = []
for t in tqdm(df["SDO_time"], desc="Matching times"):
    deltas = np.abs((times - t).total_seconds())
    idx = np.argmin(deltas)
    matched_idx.append(valid_indices[idx])

df["img_index"] = matched_idx
print("Filtered dataframe shape:", df.shape)

# load necessary images
print("loading matched HDF5 images (subset only)...")
with h5py.File(H5_PATH, "r") as f:
    images_dset = f["images"]
    X = np.empty((len(df), 512, 512), dtype=np.float32)
    for i, idx in enumerate(tqdm(df["img_index"], desc="Reading HDF5 images")):
        X[i] = images_dset[idx][...]

# clean images of nans/negative pixels
print("normalizing & reshaping images...")
X = np.nan_to_num(X, nan=0.0)
X = np.clip(X, a_min=0, a_max=None)

# global log scaling of image
X = np.log1p(X)
X_max = X.max()
X = X / X_max
X = X[..., np.newaxis]

# normalize psp footprint input
pos = df["psp_footpoint_stonyhurst_lon"].values.astype(np.float32) / 180.0
pos = pos.reshape(-1, 1)

# take in PSP distance in au
r_feature = df["psp_ephem_features_HCI_R"].values.astype(np.float32)

# concatenate both scalars into one tensor
aux_features = np.concatenate([pos, r_feature.reshape(-1, 1)], axis=1)

# log normalize the prediction targets
y_full = df[["epilo_jlinlin_offset"]].values.astype(np.float32)  # only epilo
y_log = np.log1p(y_full)
y_mean = np.mean(y_log, axis=0)
y_std = np.std(y_log, axis=0)
y = (y_log - y_mean) / y_std

# store original order for plotting
X_orig = X.copy()
y_orig = y.copy()
aux_orig = aux_features.copy()

# split training/validation sets by blocks
num_blocks = len(df) // train_block_size

# randomly shuffle blocks
rng = np.random.default_rng(seed=1717)
block_ids = np.arange(num_blocks)
rng.shuffle(block_ids)

# take first 80% of shuffled blocks in training
train_cutoff = int(0.8 * num_blocks)
train_blocks = block_ids[:train_cutoff]
val_blocks = block_ids[train_cutoff:]

# set indices of train/val sets based on selected blocks
train_idx = np.concatenate([
    np.arange(b * train_block_size, (b + 1) * train_block_size)
    for b in train_blocks
])
val_idx = np.concatenate([
    np.arange(b * train_block_size, (b + 1) * train_block_size)
    for b in val_blocks
])

# trim in case it goes over
train_idx = train_idx[train_idx < len(df)]
val_idx = val_idx[val_idx < len(df)]


# split actual data based on indices
X_train, X_val = X[train_idx], X[val_idx]
aux_train, aux_val = aux_features[train_idx], aux_features[val_idx]
y_train, y_val = y[train_idx], y[val_idx]



print(f"number of training photos: {len(train_idx)}, number of validation photos: {len(val_idx)}")

# convert to tensors as required by pytorch
X_train_t = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
X_val_t = torch.tensor(X_val.transpose(0, 3, 1, 2), dtype=torch.float32)
aux_train_t = torch.tensor(aux_train, dtype=torch.float32)
aux_val_t = torch.tensor(aux_val, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

train_ds = TensorDataset(X_train_t, aux_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, aux_val_t, y_val_t)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

print("Checking X, y, aux for NaNs/Infs...")
print("X:", np.isnan(X).sum(), "y:", np.isnan(y).sum(), "aux_features:", np.isnan(aux_features).sum())
print("X max/min:", X.max(), X.min())
print("y max/min:", y.max(), y.min())
print("aux_features max/min:", aux_features.max(), aux_features.min())

# define pytorch CNN model
class SEPModel(nn.Module):
    def __init__(self, num_conv, num_dense, dropout_rate):
        super().__init__()
        filters = [16, 32, 64, 128, 256][:num_conv]
        conv_layers = []
        in_channels = 1
        for f in filters:
            conv_layers += [
                nn.Conv2d(in_channels, f, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ]
            in_channels = f
        self.conv = nn.Sequential(*conv_layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(filters[-1] + 2, num_dense),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_dense, num_dense // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_dense // 2, 1)
        )

    def forward(self, img, aux):
        x = self.conv(img)
        x = self.global_pool(x).view(x.size(0), -1)
        x = torch.cat([x, aux], dim=1)
        return self.fc(x)

model = SEPModel(num_conv, num_dense, dropout_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=epochs,
    eta_min=1e-7
)

print(model)

# training loop
best_val_loss = float("inf")
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

    model.eval()
    val_losses = []
    with torch.no_grad():
        for Xb, auxb, yb in val_loader:
            Xb, auxb, yb = Xb.to(device), auxb.to(device), yb.to(device)
            preds = model(Xb, auxb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    scheduler.step()
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

print(f"Best validation loss: {best_val_loss:.3e}")
print(f"saved model to {MODEL_OUT}")

# evaluation and plotting
def invert_log_scaling(y_scaled):
    y_log = y_scaled * y_std + y_mean
    return np.expm1(y_log)

model.load_state_dict(torch.load(MODEL_OUT))
model.eval()

with torch.no_grad():
    y_train_pred = []
    for Xb, auxb, _ in train_loader:
        Xb, auxb = Xb.to(device), auxb.to(device)
        preds = model(Xb, auxb)
        y_train_pred.append(preds.cpu().numpy())
    y_train_pred = np.vstack(y_train_pred)

    y_val_pred = []
    for Xb, auxb, _ in val_loader:
        Xb, auxb = Xb.to(device), auxb.to(device)
        preds = model(Xb, auxb)
        y_val_pred.append(preds.cpu().numpy())
    y_val_pred = np.vstack(y_val_pred)

y_train_true_phys = invert_log_scaling(y_train)
y_val_true_phys = invert_log_scaling(y_val)
y_train_pred_phys = invert_log_scaling(y_train_pred)
y_val_pred_phys = invert_log_scaling(y_val_pred)

# scatter coloring by date
train_dates_mpl = mdates.date2num(pd.to_datetime(df.loc[train_idx, "SDO_time"]))
val_dates_mpl = mdates.date2num(pd.to_datetime(df.loc[val_idx, "SDO_time"]))

# plot actual vs. predicted
for i, name in enumerate(["epilo"]):
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

    mask_val = (y_val_true_phys[:, i] > 0) & (y_val_pred_phys[:, i] > 0)
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

# plot for specific test dates
for j, date_str in enumerate(test_dates):
    date_start = pd.Timestamp(date_str)
    date_end = date_start + pd.Timedelta(days=1)

    mask_date = (df["SDO_time"] >= date_start) & (df["SDO_time"] < date_end)
    subset_df = df[mask_date].sort_values("SDO_time")

    if len(subset_df) == 0:
        print(f"No data found for {date_str}")
        continue

    subset_idx = subset_df.index.values
    X_sub = torch.tensor(X_orig[subset_idx].transpose(0,3,1,2), dtype=torch.float32).to(device)
    aux_sub = torch.tensor(aux_orig[subset_idx], dtype=torch.float32).to(device)
    y_true_sub = y_orig[subset_idx]

    with torch.no_grad():
        y_pred_sub = model(X_sub, aux_sub).cpu().numpy()

    y_true_phys = invert_log_scaling(y_true_sub)
    y_pred_phys = invert_log_scaling(y_pred_sub)
    times_sub = pd.to_datetime(subset_df["SDO_time"])

    n = len(subset_idx)
    fig = plt.figure(figsize=(2.5 * n, 6))
    gs = gridspec.GridSpec(2, n, height_ratios=[3, 2])

    fig.suptitle(f"SEP Prediction on {date_start.date()} (3h cadence)", fontsize=14)

    for i in range(n):
        ax_img = fig.add_subplot(gs[0, i])
        img = np.expm1(X_orig[subset_idx[i], ..., 0] * X_max)
        im = ax_img.imshow(img, cmap="inferno", origin="lower")
        ax_img.set_title(times_sub.iloc[i].strftime("%H:%M"), fontsize=8)
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


# binary classification transformation
threshold = 1e-1

y_train_pred_bin = (y_train_pred_phys[:, 0] > threshold).astype(int)
y_train_true_bin = (y_train_true_phys[:, 0] > threshold).astype(int)

y_val_pred_bin = (y_val_pred_phys[:, 0] > threshold).astype(int)
y_val_true_bin = (y_val_true_phys[:, 0] > threshold).astype(int)

cm_train, acc_train, far_train, tss_train, hss_train = compute_metrics(y_train_true_bin, y_train_pred_bin)
fig_cm_train = plot_confusion_matrix(cm_train, title="Train Confusion Matrix")
wandb.log({
    "train_confusion_matrix": wandb.Image(fig_cm_train),
    "train_accuracy": acc_train,
    "train_false_alarm_rate": far_train,
    "train_tss": tss_train,
    "train_hss": hss_train
})
plt.close(fig_cm_train)

# log confusion matrix & classification statistics
cm_val, acc_val, far_val, tss_val, hss_val = compute_metrics(y_val_true_bin, y_val_pred_bin)
fig_cm_val = plot_confusion_matrix(cm_val, title="Validation Confusion Matrix")
wandb.log({
    "val_confusion_matrix": wandb.Image(fig_cm_val),
    "val_accuracy": acc_val,
    "val_false_alarm_rate": far_val,
    "val_tss": tss_val,
    "val_hss": hss_val
})
plt.close(fig_cm_val)

wandb.finish()

