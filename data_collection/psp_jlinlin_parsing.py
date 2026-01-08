# This scripts parses the Jlinlin values from the h5 files

import pandas as pd
import os

def load_epi_flux_data(h5_path, key, fields=None):
    """
    Load EPI flux data from HDF5, optionally selecting specific fields.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    
    df = pd.read_hdf(h5_path, key=key)
    
    if fields is not None:
        missing = [f for f in fields if f not in df.columns]
        if missing:
            raise ValueError(f"Fields not found in dataset: {missing}")
        df = df[fields]
    
    df = df.sort_values("time").reset_index(drop=True)
    return df

# File paths
epilo_file = "/scratch/gpfs/sk6617/ISOIS_data_Tate/spp-isois.sr.unh.edu/data_public/EPILo/epilo_Jlinlin_flux_full_mission.h5"
epihi_file = "/scratch/gpfs/sk6617/ISOIS_data_Tate/spp-isois.sr.unh.edu/data_public/EPIHi/epihi_Jlinlin_flux_full_mission.h5"

fields = ["time", "flux", "orbit"]

# Load each dataset separately
df_epilo = load_epi_flux_data(epilo_file, "epilo_data", fields=fields)
df_epihi = load_epi_flux_data(epihi_file, "epihi_data", fields=fields)

# Rename flux columns to avoid confusion
df_epilo = df_epilo.rename(columns={"flux": "flux"})
df_epihi = df_epihi.rename(columns={"flux": "flux"})

# Save each to its own CSV
df_epilo.to_csv("psp_epilo_jlinlin.csv", index=False)
df_epihi.to_csv("psp_epihi_jlinlin.csv", index=False)

print("EpiLo CSV sample:")
print(df_epilo.head())
print("\nEpiHi CSV sample:")
print(df_epihi.head())