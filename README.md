# RL-AWB: Deep Reinforcement Learning for Auto White Balance Correction in Low-Light Night-time Scenes

This repository contains inference code for our proposed RL-AWB framework.

## Requirements

```bash
pip install -r requirements.txt
```

**Required packages:**
- Python >= 3.8
- numpy==1.24.4
- scipy==1.10.1
- opencv-python==4.11.0.86
- pandas==2.2.3
- torch==2.6.0
- stable-baselines3==2.6.0
- gymnasium==1.1.1

## Quick Start

### Step 1: Download Models and Datasets

#### Option A: Automated Download (Try first)
```bash
# Install gdown
pip install gdown

# Run download script
# Windows
.\download.bat

# Linux/Mac
chmod +x download.sh
./download.sh
```

#### Option B: Manual Download

**Download Links:**
- Models (Dropbox):
  - [NCC Model](https://www.dropbox.com/scl/fi/b9sgxxeu481gr4znv5l63/NCC_model.zip?rlkey=664d2tqily9shzktn08npmqxh&st=9vt59fmj&dl=1)
  - [LEVI Model](https://www.dropbox.com/scl/fi/o6gsmpx9lpku082ybymou/LEVI_model.zip?rlkey=fbvn3e0xiu1g4q1cnjq0v0vhu&st=unqs9q6g&dl=1)
  - [Gehler Model](https://www.dropbox.com/scl/fi/2hhmgw7j597vxsmlir98t/Gehler_model.zip?rlkey=0cftcoptp9vk6abn2p7lwaa0g&st=v24ady0c&dl=1)

- Datasets (Google Drive):
  - [LEVI Dataset](https://drive.google.com/drive/folders/1VeqcIhkr83gL_ZF5DMnvFsmXyC4QhSro)
  - [NCC Dataset](https://drive.google.com/drive/folders/1MFfw-LlwNjCZz4W3NfrFC5z-SKRGl1EB) Reference: https://github.com/kaifuyang/Gray-Pixel
  - [Gehler Dataset](https://drive.google.com/drive/folders/1tlQTG3k3vu_n-IVHOI0MZV2ULn5Toi2m) Reference: https://www.cs.sfu.ca/~colour/data/shi_gehler/

See [models/README.md](models/README.md) and [dataset/README.md](dataset/README.md) for more details.

### Step 2: Run Inference

```bash
# Process entire NCC dataset
python inference.py --model_path models/NCC_model.zip --dataset NCC

# Process single image
python inference.py --model_path models/NCC_model.zip --dataset NCC --image_idx 42
```

## Usage

### Command Line Arguments

- `--model_path`: Path to the trained model file (required)
- `--dataset`: Dataset name - choose from `NCC`, `LEVI`, or `Gehler` (required)
- `--image_idx`: Index of a single image to process (optional, processes all if not specified)
- `--output_dir`: Output directory for results (default: `./results`)
- `--seed`: Random seed for reproducibility (default: 531)

### Examples

```bash
# Process entire NCC dataset
python inference.py --model_path models/NCC_model.zip --dataset NCC

# Process single image from LEVI dataset
python inference.py --model_path models/LEVI_model.zip --dataset LEVI --image_idx 100

# Custom output directory
python inference.py --model_path models/Gehler_model.zip --dataset Gehler --output_dir ./my_results

# Custom random seed
python inference.py --model_path models/NCC_model.zip --dataset NCC --seed 123
```

## Output

### Directory Structure After Running
```
results/
├── ncc_inference_detailed.csv      # Step-by-step results
└── ncc_inference_summary.txt       # Performance summary
```

### CSV Output Format

The detailed CSV file contains:
- `image_idx`: Image index
- `step`: Processing step (0 = initial, 1-100 = RL iterations)
- `arr`: Angular error (binary)
- `arr_rep`: Angular error (reproduction)
- `EvaLum_r`, `EvaLum_g`, `EvaLum_b`: Estimated illuminant RGB values
- `action0`, `action1`: Action values taken by the RL agent

### Performance Metrics

When processing the full dataset, metrics are displayed:

```
============================================================
PERFORMANCE ANALYSIS
============================================================

Start Performance (binary errors) [median, mean, trimean, best25%, worst25%]:
  2.1239, 3.1054, 2.2884, 0.6798, 7.2224

Start Performance (rep errors) [median, mean, trimean, best25%, worst25%]:
  2.9250, 4.1524, 3.0880, 0.9691, 9.5274

End Performance (binary errors) [median, mean, trimean, best25%, worst25%]:
  2.0156, 2.9847, 2.1523, 0.6234, 6.8901

End Performance (rep errors) [median, mean, trimean, best25%, worst25%]:
  2.7845, 4.0123, 2.9456, 0.8912, 9.2345

Improvement Summary:
  Improved: 312 images
  Worsened: 156 images
  Unchanged: 45 images
============================================================
```

## Expected Dataset Structure

```
dataset/
├── NCC_dataset/
│   ├── img/                   # 513 .png files
│   ├── msk/                   # 513 .png files
│   └── gt.mat
├── LEVI_dataset/
│   ├── img/                   # 700 .png files
│   ├── msk/                   # 700 .png files
│   ├── gt.mat
│   └── LEVI dataset EXIF information.csv
└── Gehler_dataset/
    ├── img/                   # 559 .png files
    ├── msk/                   # 559 .png files
    └── gt.mat
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{rlawb2026,
  title = {RL-AWB: Deep Reinforcement Learning for Auto White Balance Correction in Low-Light Night-time Scenes},
  author = {Yuan-Kang Lee and Kuan-Lin Chen and Chia-Che Chang and Yu-Lun Liu},
  booktitle = {under review},
  year = {2026},
  pages = {to appear}
}

```

