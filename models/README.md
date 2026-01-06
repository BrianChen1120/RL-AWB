# Pre-trained Models

This directory contains the pre-trained SAC (Soft Actor-Critic) models for illuminant estimation.

## Required Files

```
models/
├── NCC_model.zip
├── LEVI_model.zip
└── Gehler_model.zip
```

## Model Information

### NCC Model
- **File**: `NCC_model.zip`
- **Dataset**: NCC (513 images)
- **Algorithm**: SAC (Soft Actor-Critic)
- **Framework**: Stable-Baselines3
- **Download**: [Provide link here]
- **File Size**: [Size]
- **Trained Parameters**: 
  - Init action: [0.50, 3.5, 0.045, 0.3, 10, 3, 0, 0.90, 7, 3]
  - Adjustable params: sigma (action[1]), p (action[4])

### LEVI Model
- **File**: `LEVI_model.zip`
- **Dataset**: LEVI (700 images)
- **Algorithm**: SAC (Soft Actor-Critic)
- **Framework**: Stable-Baselines3
- **Download**: [Provide link here]
- **File Size**: [Size]
- **Trained Parameters**:
  - Init action: [0.5, 2, 0.025, 0.35, 10, 3, 0, 0.9, 7, 3]
  - Adjustable params: sigma (action[1]), p (action[4])

### Gehler Model
- **File**: `Gehler_model.zip`
- **Dataset**: Gehler (559 images)
- **Algorithm**: SAC (Soft Actor-Critic)
- **Framework**: Stable-Baselines3
- **Download**: [Provide link here]
- **File Size**: [Size]
- **Trained Parameters**:
  - Init action: [0.5, 2, 0.025, 0.35, 10, 3, 0, 0.9, 7, 3]
  - Adjustable params: sigma (action[1]), p (action[4])

## Download Instructions

1. Download the model files from the provided links
2. Place them directly in the `models/` directory
3. Ensure the file names match exactly: `NCC_model.zip`, `LEVI_model.zip`, `Gehler_model.zip`
4. Do NOT unzip the files - Stable-Baselines3 loads them directly

## Model Details

### Training Information
- **RL Algorithm**: Soft Actor-Critic (SAC)
- **Action Space**: Continuous, 2-dimensional
  - action[0]: Adjusts sigma parameter (range: -0.6 to 0.6)
  - action[1]: Adjusts p parameter (range: -4 to 4)
- **Observation Space**: 10811-dimensional
  - RGB-UV histogram features (10800-dim)
  - Action history (10-dim)
  - Current step count (1-dim)
- **Max Steps**: 100 per episode
- **Framework**: Stable-Baselines3 v2.x

### Model Performance

Expected performance metrics (angular error in degrees):

**NCC Model:**
- Start (before RL): ~3.64° (mean), ~2.39° (median)
- End (after RL): ~3.47° (mean), ~2.16° (median)

**LEVI Model:**
- Start (before RL): ~3.43° (mean), ~3.26° (median)
- End (after RL): ~3.39° (mean), ~3.18° (median)

**Gehler Model:**
- Start (before RL): [To be filled]
- End (after RL): [To be filled]

## Usage

Load a model for inference:

```bash
python inference.py --model_path models/NCC_model.zip --dataset NCC
```

The model will automatically:
1. Load the pre-trained weights
2. Initialize the environment with dataset-specific parameters
3. Run inference on the specified dataset
4. Save results and display performance metrics

## Model Compatibility

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Stable-Baselines3**: 2.0+
- **Gymnasium**: 0.28+

See `requirements.txt` in the main directory for exact versions.

## Verification

To verify models are downloaded correctly:

```bash
ls -lh models/

# Expected output:
# NCC_model.zip    [Size]
# LEVI_model.zip   [Size]
# Gehler_model.zip [Size]
```

## Troubleshooting

**Error: "Model not found"**
- Check if the file exists in `models/` directory
- Verify file name matches exactly (case-sensitive)
- Ensure file is not corrupted (re-download if necessary)

**Error: "Failed to load model"**
- Verify Stable-Baselines3 version matches training version
- Check PyTorch installation
- Ensure CUDA compatibility (CPU inference works without CUDA)

**Performance differs from reported results**
- Verify dataset is correctly prepared
- Check random seed (default: 531)
- Ensure using the correct model for the dataset

## Notes

- Models are trained with specific initialization parameters
- Cross-dataset inference (e.g., LEVI model on NCC data) is not recommended
- Models are deterministic when using the same random seed
- GPU is not required for inference (CPU works fine)

For questions or issues, please refer to the main README.md or open an issue on GitHub.