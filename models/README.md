# Pre-trained Models

This directory contains the pre-trained SAC (Soft Actor-Critic) models for illuminant estimation.

## Download Links

| Model | Dataset | Size | Download |
|-------|---------|------|----------|
| NCC_model.zip | NCC (513 images) | ~333 MB | [Dropbox](https://www.dropbox.com/scl/fi/b9sgxxeu481gr4znv5l63/NCC_model.zip?rlkey=664d2tqily9shzktn08npmqxh&st=9vt59fmj&dl=1) |
| LEVI_model.zip | LEVI (700 images) | ~333 MB | [Dropbox](https://www.dropbox.com/scl/fi/o6gsmpx9lpku082ybymou/LEVI_model.zip?rlkey=fbvn3e0xiu1g4q1cnjq0v0vhu&st=unqs9q6g&dl=1) |
| Gehler_model.zip | Gehler (559 images) | ~333 MB | [Dropbox](https://www.dropbox.com/scl/fi/2hhmgw7j597vxsmlir98t/Gehler_model.zip?rlkey=0cftcoptp9vk6abn2p7lwaa0g&st=v24ady0c&dl=1) |

## Quick Download

```bash
# Automated (using download script)
.\download.bat          # Windows
./download.sh           # Linux/Mac

# Manual (using curl)
cd models
curl -L -o NCC_model.zip "https://www.dropbox.com/scl/fi/b9sgxxeu481gr4znv5l63/NCC_model.zip?rlkey=664d2tqily9shzktn08npmqxh&st=9vt59fmj&dl=1"
curl -L -o LEVI_model.zip "https://www.dropbox.com/scl/fi/o6gsmpx9lpku082ybymou/LEVI_model.zip?rlkey=fbvn3e0xiu1g4q1cnjq0v0vhu&st=unqs9q6g&dl=1"
curl -L -o Gehler_model.zip "https://www.dropbox.com/scl/fi/2hhmgw7j597vxsmlir98t/Gehler_model.zip?rlkey=0cftcoptp9vk6abn2p7lwaa0g&st=v24ady0c&dl=1"
```

## ⚠️ Important Notes

1. **DO NOT UNZIP**: Models must remain as `.zip` files. Stable-Baselines3 loads them directly.
2. **Exact naming**: File names must match exactly: `NCC_model.zip`, `LEVI_model.zip`, `Gehler_model.zip`
3. **Correct location**: Place all models directly in the `models/` directory

## Expected Structure

```
models/
├── NCC_model.zip       # Keep as .zip
├── LEVI_model.zip      # Keep as .zip
└── Gehler_model.zip    # Keep as .zip
```

## Verification

Check if models are downloaded correctly:

```bash
# Windows (PowerShell)
dir models\*.zip

# Linux/Mac
ls -lh models/*.zip
```

Expected output:
```
NCC_model.zip      333 MB
LEVI_model.zip     333 MB
Gehler_model.zip   333 MB
```