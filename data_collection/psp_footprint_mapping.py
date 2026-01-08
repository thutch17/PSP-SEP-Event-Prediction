import pandas as pd
import numpy as np
from tqdm import tqdm
from solarmach import SolarMACH

# file paths
all_psp_file = "/Users/tate/Downloads/all_psp_data_3hr_cadence.csv"

# load data
psp_df = pd.read_csv(all_psp_file, parse_dates=["SDO_time"])

# compute SWEAP velocity magnitude if columns exist
vel_cols = ["psp_sweap_features_VEL_RTN_SUN_R",
            "psp_sweap_features_VEL_RTN_SUN_T",
            "psp_sweap_features_VEL_RTN_SUN_N"]

if all(col in psp_df.columns for col in vel_cols):
    psp_df["VSW"] = np.sqrt(
        psp_df[vel_cols[0]]**2 +
        psp_df[vel_cols[1]]**2 +
        psp_df[vel_cols[2]]**2
    )
else:
    psp_df["VSW"] = np.nan

# Default Vsw if missing
DEFAULT_VSW = 400.0

# Add column for PSP magnetic footpoint longitude if missing
if "Footprint_Lon" not in psp_df.columns:
    psp_df["Footprint_Lon"] = np.nan

# Loop over SDO timestamps with progress bar

for i, row in tqdm(psp_df.iterrows(), total=len(psp_df), desc="Computing PSP footprints"):
    breakpoint()
    date = row["SDO_time"]
    vsw = row["VSW"] if not np.isnan(row["VSW"]) else DEFAULT_VSW

    # Create SolarMACH object for PSP at this timestamp
    sm = SolarMACH(body_list=["PSP"], date=date, vsw_list=[vsw])
    coord_table = sm.coord_table

    # Extract PSP magnetic footpoint longitude
    psp_df.at[i, "Footprint_Lon"] = coord_table.iloc[0]["Magnetic footpoint longitude (Carrington)"]

# Overwrite original file with new column
psp_df.to_csv(all_psp_file, index=False)

print("PSP magnetic footpoint longitudes added to all_psp_features_sdo_timing.csv")