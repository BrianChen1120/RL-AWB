# Dataset Directory

This directory should contain the illuminant estimation datasets used for inference.

## Required Structure

```
dataset/
├── NCCdataset/
│   ├── img/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ... (up to 513.png)
│   ├── msk/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ... (up to 513.png)
│   └── gt.mat
├── LEVIdataset/
│   ├── img/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ... (up to 700.png)
│   ├── msk/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ... (up to 700.png)
│   └── gt.mat
└── Gehlerdataset/
    ├── img/
    │   ├── 1.png
    │   ├── 2.png
    │   └── ... (up to 559.png)
    ├── msk/
    │   ├── 1.png
    │   ├── 2.png
    │   └── ... (up to 559.png)
    └── gt.mat
```

## Download Instructions

### NCC Dataset
- **Download Link**: [Provide link here]
- **Number of Images**: 513
- **Description**: [Brief description]

### LEVI Dataset
- **Download Link**: [Provide link here]
- **Number of Images**: 700
- **Description**: [Brief description]

### Gehler Dataset
- **Download Link**: [Provide link here]
- **Number of Images**: 559
- **Description**: [Brief description]

## File Format Requirements

### Images (`img/` folder)
- Format: PNG
- Naming: Sequential numbers (1.png, 2.png, 3.png, ...)
- Bit depth: 14-bit for NCC and Gehler, 12-bit or 14-bit for LEVI

### Masks (`msk/` folder)
- Format: PNG (grayscale)
- Naming: Same as images (1.png, 2.png, 3.png, ...)
- Content: Binary mask (0 = masked region, 255 = valid region)

### Ground Truth (`gt.mat`)
- Format: MATLAB .mat file
- Content: Matrix of ground truth illuminant RGB values
- Variable name: `gts`, `gt`, or `real_rgb` (depending on dataset)
- Shape: (N, 3) where N is the number of images

## Verification

After downloading and organizing the datasets, verify the structure:

```bash
# Check if all required directories exist
ls -R dataset/

# Verify number of images
ls dataset/NCCdataset/img/ | wc -l    # Should be 513
ls dataset/LEVIdataset/img/ | wc -l   # Should be 700
ls dataset/Gehlerdataset/img/ | wc -l # Should be 559
```

## Notes

- All datasets must be properly organized before running inference
- Missing images or incorrect file formats will cause errors
- The code automatically handles different bit depths for LEVI dataset
- Ensure `gt.mat` files are in correct MATLAB format

## Troubleshooting

**Error: "Image not found"**
- Check if images are named correctly (1.png, not 001.png)
- Ensure PNG format (not JPG or other formats)

**Error: "Mask not found"**
- Verify mask files exist in `msk/` folder
- Check naming matches image files

**Error: "Ground truth not found"**
- Ensure `gt.mat` file exists in dataset root
- Check variable name in .mat file

For additional help, please refer to the main README.md or open an issue on GitHub.