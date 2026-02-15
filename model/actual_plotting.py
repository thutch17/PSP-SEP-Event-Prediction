import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import NullLocator, NullFormatter

# paths
PRED_DIR = "/scratch/gpfs/th5879/PSP-SEP-Event-Prediction/model/preds"
OUT_DIR  = "/scratch/gpfs/th5879/PSP-SEP-Event-Prediction/figures"
os.makedirs(OUT_DIR, exist_ok=True)

VID_BEST = "/scratch/gpfs/th5879/PSP-SEP-Event-Prediction/model/preds/VIDREG10P_val_preds_seed7.npz"
SNAP_BEST = "/scratch/gpfs/th5879/PSP-SEP-Event-Prediction/model/preds/SS10P_val_preds_seed7.npz"

VID_ALL = sorted(glob.glob(os.path.join(PRED_DIR, "VIDREG10P_val_preds_seed*.npz")))
SNAP_ALL = sorted(glob.glob(os.path.join(PRED_DIR, "SS10P_val_preds_seed*.npz")))

assert len(VID_ALL) > 0, "No video regression files found"
assert len(SNAP_ALL) > 0, "No snapshot files found"

print("Video files:", len(VID_ALL))
print("Snapshot files:", len(SNAP_ALL))

# load helpers
def load_npz(path):
    d = np.load(path)
    return d["val_true_phys"], d["val_pred_phys"], d["val_dates_mpl"]

def load_many(files):
    t, p, d = [], [], []
    for f in files:
        x = np.load(f)
        t.append(x["val_true_phys"])
        p.append(x["val_pred_phys"])
        d.append(x["val_dates_mpl"])
    return np.concatenate(t), np.concatenate(p), np.concatenate(d)

# load data
vid_best_true, vid_best_pred, vid_best_dates = load_npz(VID_BEST)
snap_best_true, snap_best_pred, snap_best_dates = load_npz(SNAP_BEST)

vid_all_true, vid_all_pred, vid_all_dates = load_many(VID_ALL)
snap_all_true, snap_all_pred, snap_all_dates = load_many(SNAP_ALL)

# compute GLOBAL axis limits so 0.1 lines align across panels
def positive_vals(a):
    return a[a > 0]

all_vals = np.concatenate([
    positive_vals(vid_best_true[:,0]),
    positive_vals(vid_best_pred[:,0]),
    positive_vals(snap_best_true[:,0]),
    positive_vals(snap_best_pred[:,0]),
    positive_vals(vid_all_true[:,0]),
    positive_vals(vid_all_pred[:,0]),
    positive_vals(snap_all_true[:,0]),
    positive_vals(snap_all_pred[:,0]),
])

global_lo = all_vals.min()
global_hi = all_vals.max()

# small padding so points don’t sit on border
pad = 1.15
global_lo /= pad
global_hi *= pad

print("Global limits:", global_lo, global_hi)

# panel plotting function
def panel(ax, ytrue, ypred, dates, target_i=0):

    mask = (ytrue[:, target_i] > 0) & (ypred[:, target_i] > 0)

    sc = ax.scatter(
        ytrue[mask, target_i],
        ypred[mask, target_i],
        c=dates[mask],
        cmap="viridis",
        s=8,
        alpha=0.7
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="box")

    lo = min(ytrue[mask, target_i].min(), ypred[mask, target_i].min())
    hi = max(ytrue[mask, target_i].max(), ypred[mask, target_i].max())

    # identical limits across all panels
    ax.set_xlim(global_lo, global_hi)
    ax.set_ylim(global_lo, global_hi)

    # 1:1 line using global limits
    ax.plot([global_lo, global_hi],
            [global_lo, global_hi],
            "r--", lw=1)

    # threshold lines
    ax.axvline(1e-1, color="red", lw=1)
    ax.axhline(1e-1, color="red", lw=1)

    # remove clutter
    ax.grid(False)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    return sc

# build figure
fig, axs = plt.subplots(2, 2, figsize=(9, 9))
axs = axs.flatten()

sc0 = panel(axs[0], vid_best_true,  vid_best_pred,  vid_best_dates)
sc1 = panel(axs[1], snap_best_true, snap_best_pred, snap_best_dates)
sc2 = panel(axs[2], vid_all_true,   vid_all_pred,   vid_all_dates)
sc3 = panel(axs[3], snap_all_true,  snap_all_pred,  snap_all_dates)

# only tick labels on top-left
for k in range(4):
    if k != 0:
        axs[k].tick_params(labelbottom=False, labelleft=False)

# shared outer labels
fig.supxlabel(r"Observed $J_{\mathrm{linlin}}$", fontsize=14)
fig.supylabel(r"Predicted $J_{\mathrm{linlin}}$", fontsize=14)

# colorbar year only
all_dates = np.concatenate([
    vid_best_dates, snap_best_dates,
    vid_all_dates, snap_all_dates
])

year_min = mdates.num2date(all_dates.min()).year
year_max = mdates.num2date(all_dates.max()).year

# force endpoints
year_min = min(year_min, 2019)
year_max = max(year_max, 2025)

years = np.arange(year_min, year_max + 1)
year_ticks = [mdates.date2num(np.datetime64(f"{y}-01-01")) for y in years]

cbar = fig.colorbar(sc3, ax=axs, shrink=0.92)
cbar.set_label("Year")
cbar.set_ticks(year_ticks)
cbar.set_ticklabels([str(y) for y in years])

# save to file
out = os.path.join(OUT_DIR, "four_panel_vid_snap_clean.png")
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)

print("saved:", out)
print("done")