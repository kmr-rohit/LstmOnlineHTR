# Online Handwriting Text Recognition (HTR) System
## Implementation Guide for Ed-Tech Tablet Application

---

## 🎯 Project Overview

Build a production-ready online handwriting recognition system for educational tablets using stroke sequence data. The system will recognize handwritten text in real-time as students write on digital tablets.

**Target Performance:**
- Character Error Rate (CER): < 10% (good), < 5% (excellent)
- Word Error Rate (WER): < 25% (good), < 15% (excellent)
- Inference Latency: < 100ms per line
- Support: Multiple languages, cursive and print writing

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Phase 1: Environment Setup](#phase-1-environment-setup)
4. [Phase 2: Data Preparation](#phase-2-data-preparation)
5. [Phase 3: Model Development](#phase-3-model-development)
6. [Phase 4: Training Pipeline](#phase-4-training-pipeline)
7. [Phase 5: Evaluation & Testing](#phase-5-evaluation--testing)
8. [Phase 6: Deployment](#phase-6-deployment)
9. [Phase 7: Production Optimization](#phase-7-production-optimization)
10. [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture Overview

### System Architecture
```
Tablet Stroke Input → Preprocessing → Model Inference → Post-processing → Text Output
                                           ↓
                                    Language Model
```

### Model Architecture (BiLSTM + CTC
```
Input: (batch_size, seq_len, 4)  # [dx, dy, dt, pen_state]
    ↓
1D Convolutional Layers (local feature extraction)
    ↓
Batch Normalization + Dropout
    ↓
Bidirectional LSTM Layers (sequential modeling)
    ↓
Fully Connected Layer
    ↓
CTC Decoder
    ↓
Output: Text String
```

**Why This Architecture?**
- ✅ Proven for online HTR (used by industry before transformers)
- ✅ Works with limited data (5k-10k samples)
- ✅ Fast inference (< 100ms)
- ✅ Easy to train and maintain
- ✅ Lower computational requirements

---

## 📁 Project Structure

```
online-htr/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config/
│   ├── config.yaml             # Main configuration
│   └── model_config.yaml       # Model hyperparameters
├── data/
│   ├── raw/                    # Raw stroke data from tablets
│   ├── processed/              # Preprocessed data
│   ├── iam_ondb/               # IAM-OnDB dataset
│   └── custom/                 # Your custom student data
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py          # Dataset classes
│   │   ├── preprocessing.py    # Stroke preprocessing
│   │   └── augmentation.py     # Data augmentation
│   ├── models/
│   │   ├── model.py            # BiLSTM+CTC model
│   │   ├── encoder.py          # CNN+LSTM encoder
│   │   └── decoder.py          # CTC decoder
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   ├── loss.py             # Loss functions
│   │   └── metrics.py          # Evaluation metrics
│   ├── inference/
│   │   ├── predictor.py        # Inference pipeline
│   │   └── lm_decoder.py       # Language model decoder
│   └── utils/
│       ├── visualization.py    # Plotting utilities
│       └── helpers.py          # Helper functions
├── scripts/
│   ├── download_iam_ondb.sh   # Download IAM dataset
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── infer.py                # Inference script
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_test.ipynb
│   └── 03_model_analysis.ipynb
├── tests/
│   └── test_*.py               # Unit tests
├── deployment/
│   ├── api/
│   │   ├── app.py              # FastAPI server
│   │   └── models.py           # API models
│   ├── docker/
│   │   └── Dockerfile
│   └── kubernetes/
│       └── deployment.yaml
├── models/                      # Saved model checkpoints
│   └── best_model.pth
└── logs/                        # Training logs
```

---

## 🚀 Phase 1: Environment Setup

### 1.1 System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- 16GB+ RAM
- 50GB disk space

### 1.2 Installation

```bash
# Clone repository
git clone <your-repo-url>
cd online-htr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 1.3 Requirements.txt

```
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Data processing
opencv-python>=4.8.0
scikit-learn>=1.3.0
pillow>=10.0.0

# Training utilities
tensorboard>=2.13.0
tqdm>=4.65.0
wandb>=0.15.0  # Optional: for experiment tracking

# Language model
pyctcdecode>=0.5.0
kenlm  # Install separately if needed

# API & Deployment
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Utilities
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 1.4 Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📊 Phase 2: Data Preparation

### 2.1 Download IAM-OnDB Dataset (Bootstrap Dataset)

```bash
# Register at: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
# Download credentials will be provided after registration

cd data/iam_ondb/
# Download the dataset (manual step)
# Extract files
unzip lineStrokes-all.tar.gz
unzip ascii-all.tar.gz
```

**IAM-OnDB Stats:**
- 13,040 text lines
- 86,272 words
- 221 writers
- English language
- Perfect for initial training

### 2.2 Understand Stroke Data Format

Your tablet should output strokes in this format:

```python
# Single stroke representation
stroke = {
    'points': [
        {'x': 120.5, 'y': 45.2, 't': 0.001, 'pen_down': True},
        {'x': 121.3, 'y': 46.1, 't': 0.015, 'pen_down': True},
        {'x': 122.1, 'y': 47.0, 't': 0.029, 'pen_down': False},  # Pen lift
        # ... more points
    ],
    'label': 'hello'  # Ground truth text
}
```

### 2.3 Preprocessing Pipeline

**Key Steps:**
1. **Normalization**: Scale coordinates to fixed height
2. **Resampling**: Uniform point density
3. **Feature Extraction**: Compute deltas (dx, dy, dt)
4. **Stroke Encoding**: Add pen state indicators

**Implementation:**

```python
# src/data/preprocessing.py

import numpy as np
from scipy.interpolate import interp1d

class StrokePreprocessor:
    def __init__(self, target_height=60, resample_density=20):
        self.target_height = target_height
        self.resample_density = resample_density
    
    def normalize(self, strokes):
        """Normalize stroke coordinates"""
        points = np.array([(p['x'], p['y']) for s in strokes for p in s['points']])
        
        # Get bounding box
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Scale to target height maintaining aspect ratio
        height = y_max - y_min
        scale = self.target_height / height if height > 0 else 1.0
        
        # Normalize
        normalized = []
        for stroke in strokes:
            norm_stroke = {
                'x': [(p['x'] - x_min) * scale for p in stroke['points']],
                'y': [(p['y'] - y_min) * scale for p in stroke['points']],
                't': [p['t'] for p in stroke['points']],
                'pen_down': [p['pen_down'] for p in stroke['points']]
            }
            normalized.append(norm_stroke)
        
        return normalized
    
    def resample(self, strokes):
        """Resample strokes to uniform point density"""
        resampled = []
        for stroke in strokes:
            # Calculate cumulative distances
            x, y = np.array(stroke['x']), np.array(stroke['y'])
            distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            cumulative = np.concatenate([[0], np.cumsum(distances)])
            
            # Resample
            total_length = cumulative[-1]
            num_points = int(total_length * self.resample_density)
            
            if num_points < 2:
                num_points = len(x)
            
            new_cumulative = np.linspace(0, total_length, num_points)
            
            # Interpolate
            fx = interp1d(cumulative, x, kind='linear')
            fy = interp1d(cumulative, y, kind='linear')
            ft = interp1d(cumulative, stroke['t'], kind='linear')
            
            resampled.append({
                'x': fx(new_cumulative).tolist(),
                'y': fy(new_cumulative).tolist(),
                't': ft(new_cumulative).tolist(),
                'pen_down': stroke['pen_down'][:num_points]
            })
        
        return resampled
    
    def extract_features(self, strokes):
        """Extract delta features (dx, dy, dt, pen_state)"""
        features = []
        
        for stroke in strokes:
            x = np.array(stroke['x'])
            y = np.array(stroke['y'])
            t = np.array(stroke['t'])
            
            # Compute deltas
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            dt = np.diff(t, prepend=t[0])
            
            # Pen state: 1 at stroke start, 0 otherwise
            pen_state = np.zeros(len(dx))
            pen_state[0] = 1
            
            # Stack features
            stroke_features = np.stack([dx, dy, dt, pen_state], axis=1)
            features.append(stroke_features)
        
        # Concatenate all strokes
        return np.vstack(features)
    
    def process(self, strokes):
        """Full preprocessing pipeline"""
        strokes = self.normalize(strokes)
        strokes = self.resample(strokes)
        features = self.extract_features(strokes)
        return features
```

### 2.4 Data Augmentation

```python
# src/data/augmentation.py

import numpy as np

class StrokeAugmentation:
    def __init__(self):
        pass
    
    def random_scale(self, features, scale_range=(0.8, 1.2)):
        """Random scaling"""
        scale = np.random.uniform(*scale_range)
        features[:, :2] *= scale  # Scale dx, dy
        return features
    
    def random_rotation(self, features, angle_range=(-15, 15)):
        """Random rotation"""
        angle = np.radians(np.random.uniform(*angle_range))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        features[:, :2] = features[:, :2] @ rotation_matrix.T
        return features
    
    def add_noise(self, features, noise_std=0.5):
        """Add Gaussian noise to coordinates"""
        noise = np.random.normal(0, noise_std, features[:, :2].shape)
        features[:, :2] += noise
        return features
    
    def time_stretch(self, features, stretch_range=(0.8, 1.2)):
        """Time stretching/compression"""
        stretch = np.random.uniform(*stretch_range)
        features[:, 2] *= stretch  # Stretch dt
        return features
    
    def augment(self, features):
        """Apply random augmentation"""
        features = self.random_scale(features.copy())
        features = self.random_rotation(features)
        features = self.add_noise(features)
        features = self.time_stretch(features)
        return features
```

### 2.5 Create Dataset Class

```python
# src/data/dataset.py

import torch
from torch.utils.data import Dataset
import json

class OnlineHTRDataset(Dataset):
    def __init__(self, data_path, preprocessor, augment=False, augmentor=None):
        self.data = self.load_data(data_path)
        self.preprocessor = preprocessor
        self.augment = augment
        self.augmentor = augmentor
        
        # Character mapping
        self.chars = self.build_char_map()
        self.char_to_idx = {c: i+1 for i, c in enumerate(self.chars)}  # 0 reserved for CTC blank
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
    
    def load_data(self, data_path):
        """Load dataset from JSON file"""
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def build_char_map(self):
        """Build character vocabulary"""
        chars = set()
        for sample in self.data:
            chars.update(sample['label'])
        return sorted(list(chars))
    
    def encode_label(self, text):
        """Encode text to indices"""
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]
    
    def decode_label(self, indices):
        """Decode indices to text"""
        return ''.join([self.idx_to_char[i] for i in indices if i in self.idx_to_char])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Preprocess strokes
        features = self.preprocessor.process(sample['strokes'])
        
        # Augment if training
        if self.augment and self.augmentor:
            features = self.augmentor.augment(features)
        
        # Encode label
        label = self.encode_label(sample['label'])
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        label = torch.LongTensor(label)
        
        return features, label, len(features), len(label)

def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    features, labels, feat_lens, label_lens = zip(*batch)
    
    # Pad sequences
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    feat_lens = torch.LongTensor(feat_lens)
    label_lens = torch.LongTensor(label_lens)
    
    return features_padded, labels_padded, feat_lens, label_lens
```

---

## 🧠 Phase 3: Model Development

### 3.1 Model Architecture

```python
# src/models/model.py

import torch
import torch.nn as nn

class OnlineHTRModel(nn.Module):
    def __init__(self, 
                 input_channels=4,
                 hidden_size=256,
                 num_layers=3,
                 num_classes=80,
                 dropout=0.3):
        super(OnlineHTRModel, self).__init__()
        
        # 1D CNN for local feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Bidirectional LSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x, lengths):
        """
        Args:
            x: (batch_size, seq_len, input_channels)
            lengths: (batch_size,) actual sequence lengths
        Returns:
            output: (batch_size, seq_len, num_classes)
        """
        batch_size = x.size(0)
        
        # Conv expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, channels)
        
        # Pack padded sequence for LSTM
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Unpack
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        # Output projection
        x = self.fc(x)
        
        return x
```

### 3.2 Model Configuration

```yaml
# config/model_config.yaml

model:
  name: "OnlineHTR_BLSTM_CTC"
  input_channels: 4  # [dx, dy, dt, pen_state]
  hidden_size: 256
  num_layers: 3
  dropout: 0.3
  
preprocessing:
  target_height: 60
  resample_density: 20
  
training:
  batch_size: 32
  num_epochs: 300
  learning_rate: 0.001
  weight_decay: 0.0001
  gradient_clip: 5.0
  
  # Learning rate scheduler
  scheduler:
    type: "ReduceLROnPlateau"
    patience: 10
    factor: 0.5
    min_lr: 0.00001
  
  # Early stopping
  early_stopping:
    patience: 20
    min_delta: 0.001

augmentation:
  enabled: true
  scale_range: [0.8, 1.2]
  rotation_range: [-15, 15]
  noise_std: 0.5
  time_stretch_range: [0.8, 1.2]

paths:
  data_dir: "data/"
  checkpoint_dir: "models/"
  log_dir: "logs/"
```

---

## 🏋️ Phase 4: Training Pipeline

### 4.1 Training Script

```python
# scripts/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import wandb  # Optional

from src.models.model import OnlineHTRModel
from src.data.dataset import OnlineHTRDataset, collate_fn
from src.data.preprocessing import StrokePreprocessor
from src.data.augmentation import StrokeAugmentation
from src.training.metrics import calculate_cer, calculate_wer

class Trainer:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize preprocessor and augmentor
        self.preprocessor = StrokePreprocessor(
            target_height=self.config['preprocessing']['target_height'],
            resample_density=self.config['preprocessing']['resample_density']
        )
        self.augmentor = StrokeAugmentation()
        
        # Load datasets
        self.train_dataset = OnlineHTRDataset(
            data_path=f"{self.config['paths']['data_dir']}/train.json",
            preprocessor=self.preprocessor,
            augment=self.config['augmentation']['enabled'],
            augmentor=self.augmentor
        )
        self.val_dataset = OnlineHTRDataset(
            data_path=f"{self.config['paths']['data_dir']}/val.json",
            preprocessor=self.preprocessor,
            augment=False
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # Initialize model
        num_classes = len(self.train_dataset.chars) + 1  # +1 for CTC blank
        self.model = OnlineHTRModel(
            input_channels=self.config['model']['input_channels'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            num_classes=num_classes,
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss function
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config['training']['scheduler']['patience'],
            factor=self.config['training']['scheduler']['factor'],
            min_lr=self.config['training']['scheduler']['min_lr']
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (features, labels, feat_lens, label_lens) in enumerate(pbar):
            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(features, feat_lens)
            
            # Prepare for CTC loss
            outputs = outputs.log_softmax(2)
            outputs = outputs.transpose(0, 1)  # (T, N, C)
            
            # Calculate loss
            loss = self.criterion(outputs, labels, feat_lens, label_lens)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for features, labels, feat_lens, label_lens in tqdm(self.val_loader, desc="Validating"):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features, feat_lens)
                outputs_log = outputs.log_softmax(2)
                outputs_log = outputs_log.transpose(0, 1)
                
                # Calculate loss
                loss = self.criterion(outputs_log, labels, feat_lens, label_lens)
                total_loss += loss.item()
                
                # Decode predictions
                outputs_np = outputs.cpu().numpy()
                for i in range(outputs_np.shape[0]):
                    pred = outputs_np[i, :feat_lens[i], :].argmax(axis=1)
                    # CTC collapse
                    pred = self.ctc_decode(pred)
                    pred_text = self.val_dataset.decode_label(pred)
                    
                    target = labels[i, :label_lens[i]].cpu().numpy()
                    target_text = self.val_dataset.decode_label(target)
                    
                    all_preds.append(pred_text)
                    all_targets.append(target_text)
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        cer = calculate_cer(all_preds, all_targets)
        wer = calculate_wer(all_preds, all_targets)
        
        return avg_loss, cer, wer
    
    def ctc_decode(self, pred):
        """Simple CTC greedy decoding"""
        result = []
        prev = -1
        for p in pred:
            if p != 0 and p != prev:  # Not blank and not repeat
                result.append(p)
            prev = p
        return result
    
    def train(self):
        print("Starting training...")
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, cer, wer = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  CER: {cer:.2f}%")
            print(f"  WER: {wer:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'cer': cer,
                    'wer': wer,
                    'config': self.config,
                    'char_to_idx': self.train_dataset.char_to_idx
                }
                
                torch.save(
                    checkpoint,
                    f"{self.config['paths']['checkpoint_dir']}/best_model.pth"
                )
                print("  ✓ Saved best model")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping']['patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\nTraining complete!")

if __name__ == "__main__":
    trainer = Trainer("config/model_config.yaml")
    trainer.train()
```

### 4.2 Metrics Calculation

```python
# src/training/metrics.py

import editdistance
import numpy as np

def calculate_cer(predictions, targets):
    """Calculate Character Error Rate"""
    total_chars = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        errors = editdistance.eval(pred, target)
        total_errors += errors
        total_chars += len(target)
    
    cer = (total_errors / total_chars) * 100 if total_chars > 0 else 0
    return cer

def calculate_wer(predictions, targets):
    """Calculate Word Error Rate"""
    total_words = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        
        errors = editdistance.eval(pred_words, target_words)
        total_errors += errors
        total_words += len(target_words)
    
    wer = (total_errors / total_words) * 100 if total_words > 0 else 0
    return wer
```

---

## 📈 Phase 5: Evaluation & Testing

### 5.1 Evaluation Script

```python
# scripts/evaluate.py

import torch
from torch.utils.data import DataLoader
import yaml
import json

from src.models.model import OnlineHTRModel
from src.data.dataset import OnlineHTRDataset, collate_fn
from src.data.preprocessing import StrokePreprocessor
from src.training.metrics import calculate_cer, calculate_wer

def evaluate_model(config_path, checkpoint_path, test_data_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize preprocessor
    preprocessor = StrokePreprocessor(
        target_height=config['preprocessing']['target_height'],
        resample_density=config['preprocessing']['resample_density']
    )
    
    # Load test dataset
    test_dataset = OnlineHTRDataset(
        data_path=test_data_path,
        preprocessor=preprocessor,
        augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    num_classes = len(checkpoint['char_to_idx'])
    model = OnlineHTRModel(
        input_channels=config['model']['input_channels'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=num_classes,
        dropout=0.0  # No dropout for inference
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate
    all_preds = []
    all_targets = []
    
    print("Evaluating...")
    with torch.no_grad():
        for features, labels, feat_lens, label_lens in test_loader:
            features = features.to(device)
            
            outputs = model(features, feat_lens)
            outputs_np = outputs.cpu().numpy()
            
            for i in range(outputs_np.shape[0]):
                pred = outputs_np[i, :feat_lens[i], :].argmax(axis=1)
                pred = ctc_decode(pred)
                pred_text = test_dataset.decode_label(pred)
                
                target = labels[i, :label_lens[i]].numpy()
                target_text = test_dataset.decode_label(target)
                
                all_preds.append(pred_text)
                all_targets.append(target_text)
    
    # Calculate metrics
    cer = calculate_cer(all_preds, all_targets)
    wer = calculate_wer(all_preds, all_targets)
    
    print(f"\nTest Results:")
    print(f"  CER: {cer:.2f}%")
    print(f"  WER: {wer:.2f}%")
    
    # Save examples
    examples = []
    for i in range(min(20, len(all_preds))):
        examples.append({
            'prediction': all_preds[i],
            'target': all_targets[i]
        })
    
    with open('test_examples.json', 'w') as f:
        json.dump(examples, f, indent=2)
    
    return cer, wer

def ctc_decode(pred):
    result = []
    prev = -1
    for p in pred:
        if p != 0 and p != prev:
            result.append(p)
        prev = p
    return result

if __name__ == "__main__":
    evaluate_model(
        config_path="config/model_config.yaml",
        checkpoint_path="models/best_model.pth",
        test_data_path="data/test.json"
    )
```

---

## 🚀 Phase 6: Deployment

### 6.1 Inference API

```python
# deployment/api/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List

from src.models.model import OnlineHTRModel
from src.data.preprocessing import StrokePreprocessor

app = FastAPI(title="Online HTR API")

# Global variables
model = None
preprocessor = None
char_map = None
device = None

class StrokePoint(BaseModel):
    x: float
    y: float
    t: float
    pen_down: bool

class RecognitionRequest(BaseModel):
    strokes: List[List[StrokePoint]]

class RecognitionResponse(BaseModel):
    text: str
    confidence: float

@app.on_event("startup")
async def load_model():
    global model, preprocessor, char_map, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    
    # Initialize preprocessor
    preprocessor = StrokePreprocessor(target_height=60, resample_density=20)
    
    # Load character mapping
    char_map = checkpoint['char_to_idx']
    idx_to_char = {i: c for c, i in char_map.items()}
    
    # Initialize model
    num_classes = len(char_map)
    model = OnlineHTRModel(
        input_channels=4,
        hidden_size=256,
        num_layers=3,
        num_classes=num_classes,
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded on {device}")

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(request: RecognitionRequest):
    try:
        # Convert request to stroke format
        strokes = []
        for stroke_points in request.strokes:
            stroke = {
                'points': [
                    {
                        'x': p.x,
                        'y': p.y,
                        't': p.t,
                        'pen_down': p.pen_down
                    }
                    for p in stroke_points
                ]
            }
            strokes.append(stroke)
        
        # Preprocess
        features = preprocessor.process(strokes)
        features = torch.FloatTensor(features).unsqueeze(0).to(device)
        feat_len = torch.LongTensor([features.size(1)])
        
        # Inference
        with torch.no_grad():
            outputs = model(features, feat_len)
            probs = torch.softmax(outputs, dim=2)
            
            # Greedy decoding
            pred = outputs[0].argmax(dim=1).cpu().numpy()
            pred = ctc_decode(pred)
            
            # Decode to text
            idx_to_char = {i: c for c, i in char_map.items()}
            text = ''.join([idx_to_char[i] for i in pred if i in idx_to_char])
            
            # Calculate confidence
            confidence = probs[0].max(dim=1)[0].mean().item()
        
        return RecognitionResponse(text=text, confidence=confidence)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "device": str(device)}

def ctc_decode(pred):
    result = []
    prev = -1
    for p in pred:
        if p != 0 and p != prev:
            result.append(p)
        prev = p
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6.2 Dockerfile

```dockerfile
# deployment/docker/Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "deployment.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.3 Client Example

```python
# Example client usage

import requests
import json

# Prepare stroke data
strokes = [
    [
        {"x": 10.0, "y": 20.0, "t": 0.0, "pen_down": True},
        {"x": 15.0, "y": 25.0, "t": 0.1, "pen_down": True},
        {"x": 20.0, "y": 30.0, "t": 0.2, "pen_down": False},
    ]
]

# Send request
response = requests.post(
    "http://localhost:8000/recognize",
    json={"strokes": strokes}
)

result = response.json()
print(f"Recognized text: {result['text']}")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## ⚡ Phase 7: Production Optimization

### 7.1 Add Language Model Decoding

```python
# src/inference/lm_decoder.py

from pyctcdecode import build_ctcdecoder
import kenlm

class LanguageModelDecoder:
    def __init__(self, vocab, lm_path=None):
        """
        Args:
            vocab: List of characters
            lm_path: Path to KenLM n-gram language model (.arpa file)
        """
        self.vocab = vocab
        
        if lm_path:
            self.decoder = build_ctcdecoder(
                labels=vocab,
                kenlm_model_path=lm_path,
            )
        else:
            self.decoder = build_ctcdecoder(labels=vocab)
    
    def decode(self, logits, beam_width=100):
        """
        Args:
            logits: (seq_len, num_classes) numpy array
            beam_width: Beam search width
        Returns:
            text: Decoded text string
        """
        text = self.decoder.decode(logits, beam_width=beam_width)
        return text

# Train language model (using KenLM)
# 1. Prepare text corpus
# 2. Run: lmplz -o 5 < corpus.txt > 5gram.arpa
# 3. Binary format: build_binary 5gram.arpa 5gram.bin
```

### 7.2 Model Quantization

```python
# scripts/quantize_model.py

import torch
from src.models.model import OnlineHTRModel

def quantize_model(checkpoint_path, output_path):
    # Load model
    checkpoint = torch.load(checkpoint_path)
    model = OnlineHTRModel(
        input_channels=4,
        hidden_size=256,
        num_layers=3,
        num_classes=len(checkpoint['char_to_idx']),
        dropout=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM},
        dtype=torch.qint8
    )
    
    # Save
    torch.save(quantized_model.state_dict(), output_path)
    print(f"Quantized model saved to {output_path}")
    
    # Compare sizes
    original_size = os.path.getsize(checkpoint_path) / 1024 / 1024
    quantized_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"Original size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Compression: {(1 - quantized_size/original_size)*100:.1f}%")
```

### 7.3 ONNX Export (for cross-platform deployment)

```python
# scripts/export_onnx.py

import torch
import torch.onnx
from src.models.model import OnlineHTRModel

def export_to_onnx(checkpoint_path, output_path):
    # Load model
    checkpoint = torch.load(checkpoint_path)
    model = OnlineHTRModel(
        input_channels=4,
        hidden_size=256,
        num_layers=3,
        num_classes=len(checkpoint['char_to_idx']),
        dropout=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 100, 4)  # (batch, seq_len, features)
    dummy_lengths = torch.LongTensor([100])
    
    # Export
    torch.onnx.export(
        model,
        (dummy_input, dummy_lengths),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input', 'lengths'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        }
    )
    print(f"Model exported to {output_path}")
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Reduce batch size or sequence length
# In config/model_config.yaml:
training:
  batch_size: 16  # Reduce from 32

# Or use gradient accumulation
accumulation_steps = 2
loss = loss / accumulation_steps
loss.backward()
if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**2. Poor CER/WER**
- Collect more training data
- Increase data augmentation
- Add language model
- Check preprocessing (visualize strokes)
- Increase model capacity (hidden_size, num_layers)

**3. Overfitting**
- Increase dropout
- Add more augmentation
- Reduce model size
- Use early stopping
- Add L2 regularization

**4. Slow Training**
- Use mixed precision training (torch.cuda.amp)
- Optimize data loading (num_workers)
- Use gradient checkpointing
- Profile code to find bottlenecks

---

## 📚 Next Steps & Improvements

### Short-term (1-3 months)
1. ✅ Get baseline BiLSTM+CTC working
2. ✅ Collect 5k+ labeled samples from students
3. ✅ Deploy MVP API
4. Add language model for better accuracy
5. Implement active learning loop

### Medium-term (3-6 months)
1. Add attention mechanism
2. Support multiple languages
3. Handle math expressions (separate recognizer)
4. Optimize for mobile/tablet deployment
5. Build data collection pipeline

### Long-term (6-12 months)
1. Experiment with Transformer architecture
2. Multi-task learning (text + math + diagrams)
3. Personalized models per student
4. Real-time collaborative features
5. Advanced error correction

---

## 📖 Resources

### Papers
- "Connectionist Temporal Classification" (Graves et al., 2006)
- "Online Handwriting Recognition with BLSTM" (Graves & Schmidhuber, 2009)
- "Deep Learning for Handwriting Recognition" (Puigcerver, 2017)

### Datasets
- IAM-OnDB: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
- HANDS-VNOnDB: Vietnamese online handwriting
- CASIA-OLHWDB: Chinese online handwriting

### Tools & Libraries
- PyTorch: https://pytorch.org
- pyctcdecode: https://github.com/kensho-technologies/pyctcdecode
- KenLM: https://kheafield.com/code/kenlm/
- FastAPI: https://fastapi.tiangolo.com

---

## 📞 Support & Contact

For questions or issues:
1. Check troubleshooting section
2. Review GitHub issues
3. Contact: [your-email@domain.com]

---

## 📄 License

[Your License Here]

---

**Happy Coding! 🚀**

Start with `scripts/train.py` and iterate from there. Remember: start simple, ship fast, improve continuously!