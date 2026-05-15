import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------------
# Paths
# -----------------------------
PRED_DIR = "/scratch/gpfs/th5879/PSP-SEP-Event-Prediction/model/preds"
OUT_DIR  = "/scratch/gpfs/th5879/PSP-SEP-Event-Prediction/figures"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load all prediction files
# -----------------------------
pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "VIDREG_val_preds_seed*.npz")))
assert len(pred_files) > 0, f"No prediction files found in {PRED_DIR}"

print(f"Found {len(pred_files)} prediction files")

train_true_all  = []
train_pred_all  = []
train_dates_all = []

val_true_all  = []
val_pred_all  = []
val_dates_all = []

seeds = []

for f in pred_files:
    print("Loading:", f)
    data = np.load(f)

    seeds.append(int(data["seed"]))

    train_true_all.append(data["train_true_phys"])
    train_pred_all.append(data["train_pred_phys"])
    train_dates_all.append(data["train_dates_mpl"])

    val_true_all.append(data["val_true_phys"])
    val_pred_all.append(data["val_pred_phys"])
    val_dates_all.append(data["val_dates_mpl"])

# concatenate across seeds
y_train_true_phys = np.concatenate(train_true_all, axis=0)
y_train_pred_phys = np.concatenate(train_pred_all, axis=0)
train_dates_mpl   = np.concatenate(train_dates_all, axis=0)

y_val_true_phys = np.concatenate(val_true_all, axis=0)
y_val_pred_phys = np.concatenate(val_pred_all, axis=0)
val_dates_mpl   = np.concatenate(val_dates_all, axis=0)

print("Combined shapes:")
print("  Train:", y_train_true_phys.shape)
print("  Val:  ", y_val_true_phys.shape)
print("  Seeds:", seeds)

# -----------------------------
# Plot actual vs predicted
# -----------------------------
TARGET_NAMES = ["epilo"]   # extend if you have more outputs

for i, name in enumerate(TARGET_NAMES):

    # =========================
    # VALIDATION
    # =========================
    mask_val = (y_val_true_phys[:, i] > 0) & (y_val_pred_phys[:, i] > 0)

    fig_val, ax_val = plt.subplots(figsize=(6, 6))
    sc = ax_val.scatter(
        y_val_true_phys[mask_val, i],
        y_val_pred_phys[mask_val, i],
        c=val_dates_mpl[mask_val],
        cmap="viridis",
        alpha=0.7,
        s=10,
    )

    lims = [
        y_val_true_phys[mask_val, i].min(),
        y_val_true_phys[mask_val, i].max(),
    ]
    ax_val.plot(lims, lims, "r--")

    ax_val.set_xscale("log")
    ax_val.set_yscale("log")
    ax_val.set_xlabel(f"Actual Jlinlin")
    ax_val.set_ylabel(f"Predicted Jlinlin")
    ax_val.set_title(f"Video Regression: Predicted vs Actual Jlinlin")
    ax_val.grid(True, which="both", ls="--")

    cbar = fig_val.colorbar(sc, ax=ax_val)
    cbar.set_label("Date")
    cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    val_out = os.path.join(OUT_DIR, f"vidreg_val_pred_vs_actual_ALL_SEEDS.png")
    fig_val.savefig(val_out, dpi=200, bbox_inches="tight")
    plt.close(fig_val)

    print("Saved:", val_out)

print("\nDone ✅")