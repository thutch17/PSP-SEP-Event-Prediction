# script to process SDO images and combine them into single h5 file

import os
import numpy as np
from astropy.io import fits
import h5py
import time
from PIL import Image
import re
import matplotlib.pyplot as plt

# downsample to 512x512 pixels
def downsample_image(image, factor=8):
    """Downsample a 2D array by averaging non-overlapping blocks."""
    h, w = image.shape
    h_trim = h - (h % factor)
    w_trim = w - (w % factor)
    image = image[:h_trim, :w_trim]
    return image.reshape(h_trim // factor, factor, w_trim // factor, factor).mean(axis=(1, 3)).astype(np.float16)


# process an individual file
def process_fits_file(filepath, downsample_factor=8):
    """Load FITS image and header, downsample, extract timestamp, optionally save PNG."""
    print('processing',filepath)
    image_data, header = fits.getdata(filepath, header=True)
    image_data = image_data.astype(np.float32)
    if downsample_factor > 1:
        image_data = downsample_image(image_data, factor=downsample_factor)

    metadata = {
        "filename": os.path.basename(filepath),
        "T_OBS": header.get("T_OBS"),
    }

    return metadata, image_data


# process a fits directory (all same one)
def process_fits_directory(input_dir, h5_path, downsample_factor=8):
    """Process all FITS files in a directory and save images + timestamps to HDF5."""
    # filter out corruped images at 21:00 exactly
    corrupted = re.compile(r"T2100\d{2}Z")
    all_files_before = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files if f.endswith(".fits")
    ]
    print("Total FITS before excluding corrupted regex:", len(all_files_before))
    print("Applying regex:", corrupted.pattern)

    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files if f.endswith(".fits") and not corrupted.search(f)
    ]

    print("Total FITS *after* excluding corrupted:", len(all_files))
    num_files = len(all_files)
    print(f"Found {num_files} FITS files")
    start_time = time.time()

    with h5py.File(h5_path, "w") as h5f:
        dset = None
        count = 0

        # metadata arrays
        dt_str = h5py.string_dtype(encoding="utf-8")
        filenames = h5f.create_dataset("filenames", (num_files,), dtype=dt_str)
        timestamps = h5f.create_dataset("T_OBS", (num_files,), dtype=dt_str)

        for i, fpath in enumerate(all_files, start=1):
            try:
                metadata, image_data = process_fits_file(fpath, downsample_factor=downsample_factor)

                if dset is None:
                    dset = h5f.create_dataset(
                        "images",
                        shape=(num_files, *image_data.shape),
                        dtype=np.float16,
                        compression="gzip",
                        compression_opts=9,
                        chunks=(1, *image_data.shape)
                    )

                # save image + metadata
                dset[count] = image_data
                filenames[count] = metadata["filename"]
                timestamps[count] = str(metadata["T_OBS"]) if metadata["T_OBS"] else ""
                count += 1

            except Exception as e:
                print(f"Error processing {fpath}: {e}")

            # progress updates
            if i % 20 == 0 or i == num_files:
                elapsed = time.time() - start_time
                avg_time_per_file = elapsed / i
                remaining = avg_time_per_file * (num_files - i)
                print(f'Progress: {i / num_files * 100:.2f}% | '
                      f'ETA: {remaining / 60:.1f} min')

    print(f"saved {count} images with timestamps to {h5_path}")


input_dir = "/scratch/gpfs/sk6617/SDO_data_Tate/AIA304/"
h5_path = "aia304_images_3hr_cadence_no_corrupted.h5"

process_fits_directory(input_dir, h5_path, downsample_factor=8)
