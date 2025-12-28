# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LSTM-based Online Handwriting Text Recognition (HTR) system for educational tablets. Uses stroke sequence data (pen coordinates over time) to recognize handwritten text in real-time.

**Architecture:** BiLSTM + CTC (Connectionist Temporal Classification)
- Input: Stroke sequences with 4 features per point: [dx, dy, dt, pen_state]
- Feature extraction: 3 1D convolutional layers
- Sequence modeling: 3-layer Bidirectional LSTM (256 hidden units)
- Output: CTC decoder for alignment-free text recognition

**Target Metrics:**
- CER (Character Error Rate): < 10% good, < 5% excellent
- WER (Word Error Rate): < 25% good, < 15% excellent
- Inference latency: < 100ms per line

## Commands

```bash
# Environment setup
pip install -r requirements.txt

# Data preparation - inspect your data format
python scripts/inspect_data.py data/db/

# Data preparation - convert and split data
python scripts/prepare_data.py convert data/db/ data/processed/
python scripts/prepare_data.py merge data/processed/ data/db/all_data.json
python scripts/prepare_data.py split data/db/all_data.json data/db/ --train 0.8 --val 0.1

# Training
python scripts/train.py --config config/model_config.yaml

# Evaluation
python scripts/evaluate.py --checkpoint models/best_model.pth --data data/db/

# Inference on single file
python scripts/infer.py --input sample.json --checkpoint models/best_model.pth
```

## Architecture

```
Tablet Stroke Input → Preprocessing → Model Inference → Post-processing → Text Output
                                            ↓
                                     Language Model (optional)
```

### Data Flow
1. Raw strokes from tablet (x, y, t, pen_state coordinates)
2. Preprocessing: normalize to fixed height, resample to uniform density, extract deltas
3. Model: Conv1D layers → BiLSTM → FC layer → CTC decoding
4. Post-processing: collapse repeated chars, remove blanks, optional LM decoding

### Key Components
- `src/data/preprocessing.py` - StrokePreprocessor: normalization, resampling, feature extraction
- `src/data/augmentation.py` - StrokeAugmentation: scaling, rotation, noise, time stretching
- `src/data/dataset.py` - OnlineHTRDataset: data loading, character mapping, collation
- `src/models/model.py` - OnlineHTRModel: BiLSTM+CTC architecture
- `src/training/metrics.py` - CER and WER calculation using edit distance
- `scripts/train.py` - Training loop with validation, checkpointing, early stopping
- `scripts/evaluate.py` - Model evaluation on test data
- `scripts/inspect_data.py` - Utility to understand stroke data format
- `scripts/prepare_data.py` - Convert/merge/split data utilities

### Stroke Data Format
```python
stroke = {
    'points': [
        {'x': 120.5, 'y': 45.2, 't': 0.001, 'pen_down': True},
        ...
    ],
    'label': 'hello'
}
```

### Processed Features
```python
features = np.array([
    [dx, dy, dt, pen_state],  # per point
    ...
])  # Shape: (seq_len, 4)
```

## Key Dependencies

- PyTorch 2.0+ (deep learning)
- scipy (stroke interpolation)
- editdistance (CER/WER metrics)
- pyctcdecode + kenlm (language model decoding)
- FastAPI + uvicorn (API deployment)

## Dataset

Primary: IAM-OnDB (requires registration at https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)
- 13,040 text lines, 86,272 words, 221 writers, English

## Model Configuration

Key hyperparameters (in `config/model_config.yaml`):
- hidden_size: 256
- num_layers: 3
- dropout: 0.3
- batch_size: 32
- learning_rate: 0.001
- gradient_clip: 5.0
- early_stopping patience: 20 epochs

## API Endpoints

- `POST /recognize` - Stroke data → recognized text + confidence
- `GET /health` - Health check
