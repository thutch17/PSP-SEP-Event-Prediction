# This file pulls together the intermediate CSV files into one dataframe, aligning
# time with the SDO image times
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


h5_path = "aia171_images_3hr_cadence.h5"
psp_csvs = [
    "psp_sweap_features.csv",
    "psp_epihi_jlinlin.csv",
    "psp_epilo_jlinlin.csv",
    "psp_ephem_features.csv",
]
output_csv = "all_psp_data_3hr_cadence.csv"
window_hours = 1.5

# load SDO timestamps
with h5py.File(h5_path, "r") as f:
    sdo_times = [t.decode("utf-8") for t in f["T_OBS"][:]]

sdo_times = pd.to_datetime(sdo_times, errors="coerce")
sdo_times = sdo_times.dropna()

# load and preprocess PSP data
psp_dfs = []
for path in psp_csvs:
    df = pd.read_csv(path)
    # detect timestamp column
    time_col = None
    for cand in ["Time", "time", "timestamp", "T_OBS", "datetime"]:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        raise ValueError(f"No timestamp column found in {path}")

    df["time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["time"])

    # merge duplicate timestamps by averaging
    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric = df_numeric.groupby(df["time"]).mean()

    # prefix column names with file basename to avoid duplicates
    prefix = os.path.splitext(os.path.basename(path))[0]
    df_numeric = df_numeric.add_prefix(prefix + "_")

    # keep time as index
    df_numeric.index.name = "time"
    psp_dfs.append(df_numeric)

# merge all PSP features on time (outer join)
psp_all = pd.concat(psp_dfs, axis=1)

# align PSP data to SDO timestamps
results = []
window = pd.Timedelta(hours=window_hours)

for t in tqdm(sdo_times, desc="Aligning PSP data with SDO times"):
    mask = (psp_all.index >= t - window) & (psp_all.index <= t + window)
    subset = psp_all.loc[mask]
    if not subset.empty:
        avg = subset.mean(numeric_only=True)
        avg["SDO_time"] = t
        results.append(avg)

# build final dataframe
final_df = pd.DataFrame(results).set_index("SDO_time")
final_df = final_df.sort_index()

# save final merged CSV
final_df.to_csv(output_csv)
print(f"saved merged features to {output_csv} with shape {final_df.shape}")