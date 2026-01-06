# RL-AWB: Deep Reinforcement Learning for Auto White Balance Correction in Low-Light Night-time Scenes

This repository contains inference code for our reinforcement learning-based illuminant estimation method.

## Dataset Preparation

Download the required datasets and organize them as follows:

```
project/
├── dataset/
│   ├── NCCdataset/
│   │   ├── img/          # Images: 1.png, 2.png, ..., 513.png
│   │   ├── msk/          # Masks: 1.png, 2.png, ..., 513.png
│   │   └── gt.mat        # Ground truth illuminants
│   ├── LEVIdataset/
│   │   ├── img/          # Images: 1.png, 2.png, ..., 700.png
│   │   ├── msk/          # Masks: 1.png, 2.png, ..., 700.png
│   │   └── gt.mat        # Ground truth illuminants
│   └── Gehlerdataset/
│       ├── img/          # Images: 1.png, 2.png, ..., 559.png
│       ├── msk/          # Masks: 1.png, 2.png, ..., 559.png
│       └── gt.mat        # Ground truth illuminants
```

### Dataset Download Links

- **NCC Dataset**: [Download Link] (Please provide link)
- **LEVI Dataset**: [Download Link] (Please provide link)
- **Gehler Dataset**: [Download Link] (Please provide link)

**Important Notes:**
- Images in `img/` folder should be PNG format (e.g., `1.png`, `2.png`, ...)
- Masks in `msk/` folder should be PNG format with same naming
- Ground truth file `gt.mat` should be in MATLAB format

## Model Download

Download the pre-trained models and place them in the `models/` directory:

```
models/
├── NCC_model.zip
├── LEVI_model.zip
└── Gehler_model.zip
```

**Model Download Links:**
- **NCC Model**: [Download Link] (Please provide link)
- **LEVI Model**: [Download Link] (Please provide link)
- **Gehler Model**: [Download Link] (Please provide link)

## Usage

### Inference on Full Dataset

Run inference on the entire dataset:

```bash
python inference.py --model_path models/NCC_model.zip --dataset NCC
```

This will:
- Process all images in the dataset
- Save detailed results to `results/ncc_inference_detailed.csv`
- Save summary to `results/ncc_inference_summary.txt`
- Display performance metrics comparing before RL (step 0) vs after RL (final step)

### Inference on Single Image

Process a specific image:

```bash
python inference.py --model_path models/NCC_model.zip --dataset NCC --image_idx 42
```

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
```

## Output Format

### CSV Output (Detailed Results)

The detailed CSV file contains step-by-step results with the following columns:
- `image_idx`: Image index
- `step`: Processing step (0 = initial, 1-100 = RL iterations)
- `arr`: Angular error (binary)
- `arr_rep`: Angular error (reproduction)
- `EvaLum_r`, `EvaLum_g`, `EvaLum_b`: Estimated illuminant RGB values
- `action0`, `action1`: Action values taken by the RL agent

### Performance Metrics

When processing the full dataset, the following metrics are displayed:

**Binary Errors:**
- Median, Mean, Trimean, Best 25%, Worst 25%

**Reproduction Errors:**
- Median, Mean, Trimean, Best 25%, Worst 25%

**Improvement Summary:**
- Number of images improved
- Number of images worsened
- Number of images unchanged

## Project Structure

```
.
├── inference.py           # Main inference script
├── env2.py               # Gymnasium environment (inference only)
├── utils/
│   ├── Algorithm.py      # Core illuminant estimation algorithm
│   └── WBsRGB.py         # RGB histogram utilities
├── models/               # Pre-trained RL models
├── dataset/              # Dataset directory (to be downloaded)
└── results/              # Inference results (auto-created)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## License

[Specify your license here]

## Contact


For questions or issues, please contact [your email] or open an issue on GitHub.
