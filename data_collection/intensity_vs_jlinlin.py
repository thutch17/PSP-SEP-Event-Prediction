import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

CSV_PATH = "/scratch/gpfs/th5879/PSP-SEP-Event-Prediction/data_collection/final_psp_df_3hr_cadence.csv"
H5_PATH = "/scratch/gpfs/th5879/PSP-SEP-Event-Prediction/data_collection/aia171_images_3hr_cadence.h5"
OUT_PATH = "/scratch/gpfs/th5879/PSP-SEP-Event-Prediction/figures/sum_intensity_vs_jlinlin.png"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# load data
print("loading dataframe...")
df = pd.read_csv(CSV_PATH)
df["SDO_time"] = pd.to_datetime(df["SDO_time"])

df = df[df["photo_captures_footprint"] != 0].reset_index(drop=True)
df = df.dropna(subset=["epilo_jlinlin_offset_10x"]).reset_index(drop=True)

J = df["epilo_jlinlin_offset_10x"].values.astype(np.float64)

# timestamps
print("loading H5 timestamps...")
with h5py.File(H5_PATH, "r") as f:
    times = np.array(f["T_OBS"], dtype=str)

times = pd.to_datetime(times, errors="coerce")
valid = ~times.isna()
valid_idx = np.where(valid)[0]
times = times[valid]

print("matching timestamps...")

matched_idx = []
for t in tqdm(df["SDO_time"], desc="matching"):
    idx = valid_idx[np.argmin(np.abs((times - t).total_seconds()))]
    matched_idx.append(idx)

df["img_index"] = matched_idx

# compute intensity sums
print("computing summed intensities...")

X_sum = np.zeros(len(df), dtype=np.float64)

with h5py.File(H5_PATH, "r") as f:
    images = f["images"]

    for i, idx in enumerate(tqdm(df["img_index"], desc="summing")):
        img = images[idx][...].astype(np.float64)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        s = img.sum()

        if not np.isfinite(s):
            s = np.nan

        X_sum[i] = s

# cleaning

print("\nremoving non-finite values...")
mask = np.isfinite(X_sum) & np.isfinite(J)

X_clean = X_sum[mask]
J_clean = J[mask]

print(f"after finite filtering: {len(X_clean)} points")

# drop outliers
print("removing outliers (log-IQR)...")

logX = np.log10(X_clean + 1e-8)

q1, q3 = np.percentile(logX, [25, 75])
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outlier_mask = (logX >= lower) & (logX <= upper)

X_final = X_clean[outlier_mask]
J_final = J_clean[outlier_mask]

print(f"after outlier removal: {len(X_final)} points retained")


# PLOT
print("plotting...")

plt.figure(figsize=(6,6))

plt.scatter(X_final, J_final, s=8, alpha=0.6)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Summed Intensity")
plt.ylabel("Jlinlin")
plt.title("Summed Intensity vs Jlinlin (outliers removed)")

plt.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=220)
plt.close()

print("saved:", OUT_PATH)
print("done")