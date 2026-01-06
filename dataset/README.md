# Dataset Directory

This directory contains the illuminant estimation datasets for inference.

## Download Links

| Dataset | Images | Size | Download |
|---------|--------|------|----------|
| NCC_dataset | 513 | ~8 GB | [Google Drive](https://drive.google.com/drive/folders/1MFfw-LlwNjCZz4W3NfrFC5z-SKRGl1EB) |
| LEVI_dataset | 700 | ~12 GB | [Google Drive](https://drive.google.com/drive/folders/1VeqcIhkr83gL_ZF5DMnvFsmXyC4QhSro) |
| Gehler_dataset | 559 | ~10 GB | [Google Drive](https://drive.google.com/drive/folders/1tlQTG3k3vu_n-IVHOI0MZV2ULn5Toi2m) |

## Quick Download

### Automated (Recommended)
```bash
# Install gdown
pip install gdown

# Run download script
.\download.bat          # Windows
./download.sh           # Linux/Mac
```

## Expected Structure

```
dataset/
├── NCC_dataset/              
│   ├── img/                  513 .png files
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ... (513.png)
│   ├── msk/                  513 .png files
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ... (513.png)
│   └── gt.mat
│
├── LEVI_dataset/             
│   ├── img/                  700 .png files
│   ├── msk/                  700 .png files
│   ├── gt.mat
│   └── LEVI dataset EXIF information.csv
│
└── Gehler_dataset/
    ├── img/                  559 .png files
    ├── msk/                  559 .png files
    └── gt.mat
```
