# This script loads all of the PSP data from the cdf files into csv intermediates

import cdflib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
import astropy.units as u
from astropy.time import Time
import spiceypy as spice

# load spice for timing
spice_kernel = '/scratch/gpfs/sk6617/naif0012.tls'
spice.furnsh(spice_kernel)


# take in path to cdf file, list of features, return dataframe with row of features for each epoch value
def load_cdf_to_df(path, features):
    cdf = cdflib.CDF(path)
    epochs = np.array(cdf.varget("Epoch"))

    # convert CDF epochs to datetime objects
    human_time = cdflib.epochs.CDFepoch().to_datetime(epochs)

    # round to nearest second
    human_time = pd.Series(human_time).dt.round("1s")

    data = {"Time": human_time}

    available_vars = cdf.cdf_info().zVariables

    for f in features:
        if f == "Epoch":
            continue
        if f not in available_vars:
            continue
        values = np.array(cdf.varget(f))
        # case of VEL_RTN_SUN tuple, need to unpack R, T, N values
        if values.ndim == 2 and values.shape[1] == 3:
            data[f + "_R"] = values[:, 0]
            data[f + "_T"] = values[:, 1]
            data[f + "_N"] = values[:, 2]
        else:
            data[f] = values

    dataframe = pd.DataFrame(data)

    if all(col in dataframe.columns for col in ["HCI_R", "HCI_Lon", "HCI_Lat"]):
        times = Time(dataframe["Time"])

        hci_coord = SkyCoord(
            lon=dataframe["HCI_Lon"].values * u.deg,
            lat=dataframe["HCI_Lat"].values * u.deg,
            distance=dataframe["HCI_R"].values * u.au,
            frame=frames.HeliocentricInertial(obstime=times),
            representation_type="spherical",
        )

        hgs_coord = hci_coord.transform_to(frames.HeliographicStonyhurst(obstime=times))
        dataframe["HGS_Lon"] = hgs_coord.lon.to(u.deg).value

    # drop rows w/o any features
    dataframe = dataframe.dropna(how="all", subset=[col for col in dataframe.columns if col != "Time"])
    dataframe = dataframe.drop_duplicates(subset=["Time"], keep="first")

    return dataframe


# take in directory, list of features, and optionally filter keyword to filter cdf files, and recursively return single dataframe
# with all features in the directory
def load_cdf_directory(directory, features, filter_keyword=""):
    dfs = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".cdf") and filter_keyword in fname:
                path = os.path.join(root, fname)
                df = load_cdf_to_df(path, features)
                if df is not None:
                    dfs.append(df)

    dataframe = pd.concat(dfs)
    dataframe = dataframe.sort_values("Time").reset_index(drop=True)
    dataframe = dataframe.drop_duplicates(subset=["Time"], keep="first")  # <- added
    return dataframe


# plot feature across all epochs for a given feature/dataframe
def plot_feature(df, feature, title=None):
    if feature not in df.columns:
        print(f"Feature '{feature}' not found in DataFrame.")
        return
    
    plt.figure(figsize=(10, 4))
    plt.plot(df['Time'], df[feature], marker='.', linestyle='None', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel(feature)
    plt.title(title if title else feature)
    plt.grid(True)
    plt.xticks(rotation=30)
    plt.tight_layout()
    
    filename = f"{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()


# directories
psp_ephem_dir = '/scratch/gpfs/sk6617/ISOIS_data_Tate/spp-isois.sr.unh.edu/data_public/ISOIS/level2'
psp_sweap_dir = '/scratch/gpfs/sk6617/ISOIS_data_Tate/sweap.cfa.harvard.edu/pub/data/sci/sweap/spi/L3/spi_sf00/'

ephem_features = ['Epoch', 'HCI_R', 'HCI_Lat', 'HCI_Lon']
sweap_features = ['Epoch', 'VEL_RTN_SUN']

# load CDF directories
df_ephem = load_cdf_directory(psp_ephem_dir, ephem_features, "ephem")
df_ephem.to_csv("psp_ephem_features.csv", index=False)
print('Saved ephem features')

df_sweap = load_cdf_directory(psp_sweap_dir, sweap_features)
df_sweap.to_csv("psp_sweap_features.csv", index=False)
print('Saved sweap features')

print("Ephem:")
print(df_ephem.head())
print(f"Total rows: {len(df_ephem)}\n")

print("SWEAP:")
print(df_sweap.head())
print(f"Total rows: {len(df_sweap)}")