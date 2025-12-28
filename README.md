# LSTM Online Handwriting Text Recognition

A PyTorch implementation of online handwriting recognition using BiLSTM + CTC architecture, trained on the IAM-OnDB dataset.

## Overview

This system recognizes handwritten text from stroke sequences (pen coordinates over time) captured on digital tablets or whiteboards. Unlike offline HTR which works with images, online HTR uses the temporal stroke information for improved accuracy.

**Architecture:** 1D CNN → Bidirectional LSTM → CTC Decoder

**Dataset:** IAM Online Handwriting Database (12,179 text lines, 221 writers)

## Features

- BiLSTM encoder with 1D CNN feature extraction
- CTC loss for alignment-free training
- Data augmentation (scaling, rotation, noise, time stretching)
- Configurable hyperparameters via YAML
- Train/validation/test split support
- Checkpoint saving and early stopping

## Project Structure

```
LstmOnlineHTR/
├── config/
│   └── model_config.yaml      # Training configuration
├── data/
│   ├── db/                    # Raw IAM-OnDB data
│   │   ├── lineStrokes/       # Stroke XML files
│   │   ├── labels.mlf         # Transcriptions
│   │   └── *.txt              # Train/test splits
│   └── processed/             # Parsed JSON files
├── scripts/
│   ├── parse_iam_ondb.py      # Convert XML to JSON
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── infer.py               # Single-file inference
│   └── inspect_data.py        # Data inspection utility
├── src/
│   ├── data/
│   │   ├── preprocessing.py   # Stroke preprocessing
│   │   ├── augmentation.py    # Data augmentation
│   │   └── dataset.py         # PyTorch dataset
│   ├── models/
│   │   └── model.py           # BiLSTM+CTC model
│   └── training/
│       └── metrics.py         # CER/WER metrics
├── models/                    # Saved checkpoints
├── logs/                      # Training logs
├── requirements.txt
├── data_readme.md             # Dataset documentation
└── README.md
```

## Quick Start

### 1. Installation

```bash
git clone https://github.com/kmr-rohit/LstmOnlineHTR.git
cd LstmOnlineHTR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

The repository includes IAM-OnDB data. Parse it to JSON format:

```bash
python scripts/parse_iam_ondb.py --data-dir data/db --output-dir data/processed
```

### 3. Train

```bash
python scripts/train.py --config config/model_config.yaml
```

### 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint models/best_model.pth --data data/processed/test.json
```

## Commands Reference

### Data Preparation

```bash
# Inspect raw data format
python scripts/inspect_data.py data/db/lineStrokes/

# Parse IAM-OnDB to JSON
python scripts/parse_iam_ondb.py --data-dir data/db --output-dir data/processed

# Check processed data
python scripts/inspect_data.py data/processed/train.json
```

### Training

```bash
# Train with default config
python scripts/train.py

# Train with custom config
python scripts/train.py --config config/model_config.yaml

# Train in background
nohup python scripts/train.py > logs/training.log 2>&1 &

# Monitor training
tail -f logs/training.log
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint models/best_model.pth \
    --data data/processed/test.json \
    --output test_results.json

# Evaluate on validation set
python scripts/evaluate.py \
    --checkpoint models/best_model.pth \
    --data data/processed/val.json
```

### Inference

```bash
# Run inference on single sample
python scripts/infer.py \
    --checkpoint models/best_model.pth \
    --input sample.json
```

### Utilities

```bash
# Check GPU usage
nvidia-smi

# Watch GPU in real-time
watch -n 1 nvidia-smi

# Find running training processes
ps aux | grep train.py

# Kill training process
pkill -f "python scripts/train.py"
```

## Configuration

Edit `config/model_config.yaml`:

```yaml
model:
  hidden_size: 256      # LSTM hidden units
  num_layers: 3         # LSTM layers
  dropout: 0.3          # Dropout rate

training:
  batch_size: 32
  learning_rate: 0.0003
  gradient_clip: 1.0    # Gradient clipping threshold
  num_epochs: 300

  scheduler:
    patience: 10        # LR reduction patience
    factor: 0.5         # LR reduction factor

  early_stopping:
    patience: 30        # Early stopping patience

augmentation:
  enabled: true
  scale_range: [0.9, 1.1]
  rotation_range: [-10, 10]
  noise_std: 0.3
```

## Model Architecture

```
Input: (batch, seq_len, 4)     # [dx, dy, dt, pen_state]
    ↓
Conv1D(4→64) + BN + ReLU + Dropout
Conv1D(64→128) + BN + ReLU + Dropout
Conv1D(128→256) + BN + ReLU + Dropout
    ↓
BiLSTM(256→256×2) × 3 layers
    ↓
Linear(512→num_classes)
    ↓
CTC Decoder
    ↓
Output: Text string
```

**Parameters:** ~4.4M trainable

## Performance Targets

| Metric | Good | Excellent |
|--------|------|-----------|
| CER (Character Error Rate) | < 10% | < 5% |
| WER (Word Error Rate) | < 25% | < 15% |
| Inference Latency | < 100ms | < 50ms |

## Dataset

See [data_readme.md](data_readme.md) for detailed dataset documentation.

**IAM-OnDB Statistics:**
- 12,179 text lines
- 54 unique characters
- Average label length: 32 characters
- Train/Val/Test split: 4,838 / 537 / 6,804

## Troubleshooting

### Training instability (loss explosion)
```yaml
# Reduce learning rate and increase gradient clipping
training:
  learning_rate: 0.0001
  gradient_clip: 0.5
```

### Out of memory
```yaml
# Reduce batch size
training:
  batch_size: 16
```

### Slow training
```yaml
# Increase workers for data loading
training:
  num_workers: 4
  batch_size: 64  # if GPU memory allows
```

### GPU not detected
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
CUDA_VISIBLE_DEVICES="" python scripts/train.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory
- 16GB+ RAM

## License

MIT License

## References

- [IAM-OnDB Dataset](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)
- [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) - Graves et al., 2006
- [Online Handwriting Recognition with BLSTM](https://www.cs.toronto.edu/~graves/nips_2008.pdf) - Graves & Schmidhuber, 2008
