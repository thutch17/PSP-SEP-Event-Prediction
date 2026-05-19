# Solar Energetic Particle Prediction in the Inner Heliosphere Using Deep Learning and PSP/IS☉IS Data

Github repository to house the data collection and model code for the research paper "Solar Energetic Particle Prediction in the Inner Heliosphere Using Deep Learning and PSP/IS☉IS Data" for publication in Journal of Geophysical Research - Machine Learning and Computation.


All code was run on Princeton University's Stellar compute cluster. The large quantity of Solar Dynamics Observatory (SDO) images for each wavelength are too large to be stored in this repository, but [here](https://drive.google.com/drive/folders/1op7nG-XlHzoQqb1v0J_XMF7LqGV5tsqv?usp=sharing). 


As outlined in the paper, there are 4 models: snapshot regression (model/snapshotregressionmodel.py), which predicts the output J_{linlin} off of a single image, snapshot classification (model/snapshotclassificationmodel.py), which classifies a single image as a SEP event or not, video regression (model/videoregressionmodel.py), which predicts J_{linlin} off of a sliding window of images, and video classification (model/videoclassificationmodel.py), which classifies a sliding window of images as a SEP event or not.

---

## Prerequisites

All models require two data files:

1. **Tabular data (CSV):** `data_collection/final_psp_df_3hr_cadence.csv`
    - Available in this repository
2. **Image data (HDF5):** `data_collection/aia{wavelength}_images_3hr_cadence.h5`  
   Replace `{wavelength}` with the SDO/AIA wavelength you intend to use (94, 131, 171, 193, 211, 304, or 335). For example, the default wavelength of 171 Å requires `aia171_images_3hr_cadence.h5`.
   - Available under the previous google drive link, place in the `data_collection` directory


The video models additionally cache preprocessed images to disk on first run as `data_collection/matched_images_{wavelength}.pkl`. This file is created automatically to avoid preprocessing images multiple times.

---

## Running the Models

All model scripts are in the `model/` directory. Run them from that directory:

```bash
cd model/
```

Trained model weights are saved to `model/models/` and predictions to `model/preds/`.

---

### 1. Snapshot Regression (`snapshotregressionmodel.py`)

Predicts continuous J_linlin flux from a single SDO/AIA image.

```bash
python snapshotregressionmodel.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--wavelength` | `171` | SDO/AIA wavelength in Å (94, 131, 171, 193, 211, 304, 335) |
| `--epochs` | `500` | Number of training epochs (early stopping almost always is the limiter) |
| `--batch_size` | `32` | Mini-batch size |
| `--learning_rate` | `0.0003` | Adam learning rate |
| `--num_dense_nodes` | `64` | Neurons in the fully connected head |
| `--num_conv` | `5` | Number of convolutional layers |
| `--dropout` | `0.25` | Dropout probability |
| `--train_block_size` | `80` | Number of consecutive samples per training block |
| `--train_fraction` | `1` | Fraction of each training block to use (0–1) |
| `--seed` | `1717` | Random seed for train/test split |

**Example:**
```bash
python snapshotregressionmodel.py --wavelength=171 --seed=1717
```

---

### 2. Snapshot Classification (`snapshotclassificationmodel.py`)

Binary classification (SEP event or not) from a single SDO/AIA image. Events are defined by a J_linlin threshold of 1e-1.

```bash
python snapshotclassificationmodel.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--wavelength` | `171` | SDO/AIA wavelength in Å (94, 131, 171, 193, 211, 304, 335) |
| `--epochs` | `500` | Number of training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--learning_rate` | `0.0003` | Adam learning rate |
| `--num_dense_nodes` | `64` | Neurons in the fully connected head |
| `--num_conv` | `5` | Number of convolutional layers |
| `--dropout` | `0.25` | Dropout probability |
| `--train_block_size` | `80` | Number of consecutive samples per training block |
| `--train_fraction` | `1` | Fraction of each training block to use (0–1) |
| `--seed` | `1717` | Random seed for train/test split |

**Example:**
```bash
python snapshotclassificationmodel.py --wavelength=171 --seed=1717
```

To run with multiple seeds for ensemble evaluation:
```bash
for seed in 1717 7 1337 42 123; do
    python snapshotclassificationmodel.py --seed=$seed
done
```

---

### 3. Video Regression (`videoregressionmodel.py`)

Predicts continuous J_linlin flux from a sliding window of sequential SDO/AIA images using a ResNet18 + Transformer encoder architecture.

```bash
python videoregressionmodel.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--wavelength` | `171` | SDO/AIA wavelength in Å (94, 131, 171, 193, 211, 304, 335) |
| `--epochs` | `500` | Number of training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--learning_rate` | `0.0001` | Adam learning rate |
| `--dropout` | `0.25` | Dropout probability |
| `--window_size` | `8` | Number of frames in the temporal sliding window |
| `--n_heads` | `4` | Number of transformer attention heads |
| `--n_attn_blocks` | `3` | Number of transformer encoder blocks |
| `--hidden_head` | `256` | Transformer hidden dimension |
| `--train_block_size` | `80` | Number of consecutive samples per training block |
| `--train_fraction` | `1` | Fraction of each training block to use (0–1) |
| `--seed` | `1717` | Random seed for train/test split |

The ResNet18 backbone weights are loaded from `model/resnet18_imagenet.pth`. Preprocessed images are cached to `data_collection/matched_images_{wavelength}.pkl` on first run.

**Example:**
```bash
python videoregressionmodel.py --wavelength=171 --window_size=8 --seed=1717
```

---

### 4. Video Classification (`videoclassificationmodel.py`)

Binary classification (SEP event or not) from a sliding window of sequential SDO/AIA images using the same ResNet18 + Transformer encoder architecture as video regression.

```bash
python videoclassificationmodel.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--wavelength` | `171` | SDO/AIA wavelength in Å (94, 131, 171, 193, 211, 304, 335) |
| `--epochs` | `500` | Number of training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--learning_rate` | `0.0001` | Adam learning rate |
| `--window_size` | `8` | Number of frames in the temporal sliding window |
| `--n_heads` | `4` | Number of transformer attention heads |
| `--n_attn_blocks` | `3` | Number of transformer encoder blocks |
| `--hidden_head` | `256` | Transformer hidden dimension |
| `--seed` | `1717` | Random seed for train/test split |

**Example:**
```bash
python videoclassificationmodel.py --wavelength=171 --window_size=8 --seed=1717
```

To explore different wavelengths and architectures:
```bash
python videoclassificationmodel.py --wavelength=94 --n_heads=8 --n_attn_blocks=4
python videoclassificationmodel.py --wavelength=211 --n_heads=4 --n_attn_blocks=3
```

---

## Output

Each model run produces:
- **Trained weights:** `model/models/{name}.pt` where `{name}` encodes the hyperparameters used.
- **Predictions:** `model/preds/{name}.npz` containing validation set predictions and ground truth.
- **Metrics and plots:** Logged locally via Weights & Biases in offline mode (`WANDB_MODE=offline`).

Classification models report accuracy, precision, recall (POD), false alarm ratio, TSS, HSS, and F1. Regression models report MSE and generate scatter and time series plots.

For any questions, please reach out to tatehutchins@princeton.edu!