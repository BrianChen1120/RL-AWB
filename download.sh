#!/bin/bash

set -e

echo "=========================================="
echo "  Downloading Models and Datasets"
echo "=========================================="
echo ""

# Create directories
mkdir -p models
mkdir -p dataset

# Download models from Dropbox
echo "[1/6] Downloading NCC model..."
wget -O models/NCC_model.zip "https://www.dropbox.com/scl/fi/b9sgxxeu481gr4znv5l63/NCC_model.zip?rlkey=664d2tqily9shzktn08npmqxh&st=9vt59fmj&dl=1"

echo "[2/6] Downloading LEVI model..."
wget -O models/LEVI_model.zip "https://www.dropbox.com/scl/fi/o6gsmpx9lpku082ybymou/LEVI_model.zip?rlkey=fbvn3e0xiu1g4q1cnjq0v0vhu&st=unqs9q6g&dl=1"

echo "[3/6] Downloading Gehler model..."
wget -O models/Gehler_model.zip "https://www.dropbox.com/scl/fi/2hhmgw7j597vxsmlir98t/Gehler_model.zip?rlkey=0cftcoptp9vk6abn2p7lwaa0g&st=v24ady0c&dl=1"

# Download datasets from Google Drive
echo "[4/6] Downloading LEVI_dataset from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/1VeqcIhkr83gL_ZF5DMnvFsmXyC4QhSro" -O dataset/LEVI_dataset --remaining-ok

echo "[5/6] Downloading NCC_dataset from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/1MFfw-LlwNjCZz4W3NfrFC5z-SKRGl1EB" -O dataset/NCC_dataset --remaining-ok

echo "[6/6] Downloading Gehler_dataset from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/1tlQTG3k3vu_n-IVHOI0MZV2ULn5Toi2m" -O dataset/Gehler_dataset --remaining-ok

echo ""
echo "=========================================="
echo "  Download Complete!"
echo "=========================================="
echo ""
echo "Directory structure:"
echo "  models/NCC_model.zip"
echo "  models/LEVI_model.zip"
echo "  models/Gehler_model.zip"
echo "  dataset/NCC_dataset/"
echo "  dataset/LEVI_dataset/"
echo "  dataset/Gehler_dataset/"
echo ""
echo "Next steps:"
echo "  pip install -r requirements.txt"
echo "  python test_imports.py"
echo "  python inference.py --model_path models/NCC_model.zip --dataset NCC"
echo ""