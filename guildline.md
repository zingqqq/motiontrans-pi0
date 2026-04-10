# Guideline for Running MotionTrans

This document provides step-by-step instructions to preprocess the dataset, compute normalization statistics, and train the MotionTrans model.

---

## 1. Dataset Preprocessing

1. Navigate to:

```bash
cd motiontrans/scripts_data
```

2. Edit the script:

```bash
zarr_human(robot)_data_conversion_batch.sh
```

Modify the following fields:

- `input_dir`: Path to your raw dataset

- `resolution`: Adjust according to your data

>  The folder naming under `input_dir` should follow the convention used in the official repository.

3. Run preprocessing (for human and robot data separately):

```bash
cd motiontrans
bash ./scripts_data/zarr_human_data_conversion_batch.sh
bash ./scripts_data/zarr_robot_data_conversion_batch.sh
```

---

## 2. Compute Normalization (Co-train)

1. Navigate to:

```bash
cd motiontrans-pi0/scripts_exp
```

2. Edit:

```bash
get_normalize_cotrain.sh
```

Modify:

- `dataset_path`: Path to preprocessed zarr datasets

Example:

```bash
/data/xxx/zarr_data_human|/data/xxx/zarr_data_robot
```

> Use `|` to concatenate multiple datasets.

- `CUDA device ID`
3. Run:

```bash
cd motiontrans-pi0
bash ./scripts_exp/get_normalize_cotrain.sh
```

---

## 3. Training Pipeline

1. Navigate to:

```bash
cd motiontrans-pi0/scripts_exp
```

2. Edit:

```bash
train_cotrain.sh
```

Modify:

- `dataset_path` (same format as above)

- `CUDA device ID`

- Other hyperparameters (e.g., `batch_size`)
3. Run training:

```bash
cd motiontrans-pi0
bash ./scripts_exp/train_cotrain.sh
```

---

## Notes

- **Weights & Biases (wandb)**  
  Training uses `wandb` for logging. Please make sure it is set up before training:

```bash
wandb login
```

- **Visualization (Optional)**

To enable visualization during preprocessing:

1. Open:

```
motiontrans-pi0/src/openpi/policies/dataset_zarr.py
```

2. Uncomment the visualization code at the end of the file

>  Disable (comment out) visualization before running training, as it may slow down or interfere with training.


