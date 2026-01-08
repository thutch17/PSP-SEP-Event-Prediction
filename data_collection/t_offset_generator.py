import pandas as pd
from tqdm import tqdm

AU_TO_KM = 1.496e8
SPEED = 3 * 400.0

final_df = pd.read_csv("final_psp_df_3hr_cadence.csv", parse_dates=["SDO_time"])
epihi_df = pd.read_csv("psp_epihi_jlinlin.csv", parse_dates=["time"])
epilo_df = pd.read_csv("psp_epilo_jlinlin.csv", parse_dates=["time"])

# ensure sorted times
epihi_df = epihi_df.sort_values("time")
epilo_df = epilo_df.sort_values("time")

# add t_offset column
if "t_offset_s" not in final_df.columns:
    final_df["t_offset_s"] = (final_df["psp_ephem_features_HCI_R"] * AU_TO_KM) / SPEED

offset = pd.to_timedelta(final_df["t_offset_s"], unit="s")

# helper to compute mean flux over the given window
def compute_mean_flux(df, start, end):
    mask = (df["time"] >= start) & (df["time"] <= end)
    if not mask.any():
        return np.nan
    return df.loc[mask, "flux"].mean()

# Prepare output columns
epihi_vals = []
epilo_vals = []

for SDO_time, dt in tqdm(zip(final_df["SDO_time"], offset), total=len(final_df), desc="Processing rows"):
    t_start = SDO_time + dt
    t_end = t_start + pd.Timedelta(hours=12)

    epihi_vals.append(compute_mean_flux(epihi_df, t_start, t_end))
    epilo_vals.append(compute_mean_flux(epilo_df, t_start, t_end))

final_df["epihi_jlinlin_offset"] = epihi_vals
final_df["epilo_jlinlin_offset"] = epilo_vals

final_df.to_csv("final_psp_df_3hr_cadence_with_offsets.csv", index=False)

print("done! new columns created using per-row t_offset.")