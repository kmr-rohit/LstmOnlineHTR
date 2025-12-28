# IAM-OnDB Dataset Guide

This document explains the IAM Online Handwriting Database (IAM-OnDB) structure, storage format, and preprocessing pipeline used in this project.

## Dataset Overview

**IAM-OnDB** is a benchmark dataset for online handwriting recognition containing:
- **12,179 text lines** with stroke sequences
- **221 writers**
- **English language** text from LOB corpus
- **Stroke-level** coordinate data (x, y, time)

**Source:** https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database

## Directory Structure

```
data/db/
├── lineStrokes/                    # Stroke XML files
│   ├── a01/                        # Writer/form groups
│   │   ├── a01-000/
│   │   │   ├── a01-000u-01.xml    # Individual line strokes
│   │   │   ├── a01-000u-02.xml
│   │   │   └── ...
│   │   └── a01-001/
│   │       └── ...
│   ├── a02/
│   └── ...
├── labels.mlf                      # Master Label File (transcriptions)
├── trainset.txt                    # Training set form IDs
├── testset_f.txt                   # Test set (writer-independent)
├── testset_t.txt                   # Test set (text-independent)
└── testset_v.txt                   # Validation set
```

## File Formats

### 1. Stroke XML Files (`lineStrokes/*.xml`)

Each XML file contains stroke data for one text line:

```xml
<?xml version="1.0" encoding="ISO-8859-1"?>
<WhiteboardCaptureSession>
  <WhiteboardDescription>
    <SensorLocation corner="top_left"/>
    <DiagonallyOppositeCoords x="6537" y="5609"/>
  </WhiteboardDescription>
  <StrokeSet>
    <Stroke colour="black" start_time="14978711.86" end_time="14978712.28">
      <Point x="674" y="5360" time="14978711.86"/>
      <Point x="672" y="5356" time="14978711.88"/>
      <Point x="673" y="5353" time="14978711.90"/>
      <!-- More points... -->
    </Stroke>
    <Stroke colour="black" start_time="14978712.50" end_time="14978712.79">
      <!-- Next stroke (after pen lift) -->
    </Stroke>
  </StrokeSet>
</WhiteboardCaptureSession>
```

**Key elements:**
- `<Stroke>`: One continuous pen-down segment
- `<Point>`: Individual coordinate with x, y, and timestamp
- Multiple strokes = pen was lifted between them

### 2. Labels File (`labels.mlf`)

HTK Master Label File format containing character-level transcriptions:

```
#!MLF!#
"/path/to/a01-000u-01.lab"
A
sp
M
O
V
E
sp
t
o
sp
...
.
"/path/to/a01-000u-02.lab"
...
```

**Special tokens:**
- `sp` = space character
- `ga` = garbage/unknown (skipped during parsing)
- Single characters = literal characters

### 3. Split Files (`trainset.txt`, `testset_*.txt`)

Plain text files with form IDs, one per line:

```
a01-020x
d04-081
f02-000
...
```

## File Naming Convention

```
a01-000u-01.xml
│││ │││ │  └── Line number (01, 02, ...)
│││ │││ └───── Form variant (u, w, x, z, or none)
│││ └──────── Form number (000, 001, ...)
└──────────── Writer ID (a01, b02, ...)
```

## Preprocessing Pipeline

### Step 1: Parse Raw Data

Run the parser to convert XML + MLF to JSON:

```bash
python scripts/parse_iam_ondb.py --data-dir data/db --output-dir data/processed
```

**Output:** Creates `train.json`, `val.json`, `test.json` in `data/processed/`

### Step 2: JSON Data Format

Each sample in the JSON files:

```json
{
  "id": "a01-000u-01",
  "strokes": [
    {
      "points": [
        {"x": 674.0, "y": 5360.0, "t": 14978711.86, "pen_down": true},
        {"x": 672.0, "y": 5356.0, "t": 14978711.88, "pen_down": true},
        ...
      ]
    },
    {
      "points": [...]  // Next stroke after pen lift
    }
  ],
  "label": "A MOVE to stop Mr. Gaitskell from"
}
```

### Step 3: Feature Extraction

The `StrokePreprocessor` class performs:

#### 3.1 Normalization
- Scale coordinates to fixed height (default: 60 units)
- Maintain aspect ratio
- Translate to origin (0, 0)

```python
# Before: x ∈ [500, 2000], y ∈ [5000, 5200]
# After:  x ∈ [0, 450], y ∈ [0, 60]
```

#### 3.2 Resampling
- Remove duplicate consecutive points
- Resample to uniform point density along arc length
- Default: 20 points per unit length

#### 3.3 Delta Feature Extraction
Convert absolute coordinates to deltas:

```python
# Input: (x, y, t) absolute coordinates
# Output: (dx, dy, dt, pen_state) per point

features = [
    [dx, dy, dt, pen_state],  # Point 1
    [dx, dy, dt, pen_state],  # Point 2
    ...
]
# Shape: (sequence_length, 4)
```

**Feature meanings:**
- `dx`: Change in x from previous point
- `dy`: Change in y from previous point
- `dt`: Time delta from previous point
- `pen_state`: 1 if stroke start (pen was lifted), 0 otherwise

### Step 4: Data Augmentation (Training Only)

Applied randomly during training:

| Augmentation | Range | Description |
|--------------|-------|-------------|
| Scaling | 0.9 - 1.1 | Scale dx, dy uniformly |
| Rotation | -10° to +10° | Rotate stroke coordinates |
| Noise | σ = 0.3 | Gaussian noise on dx, dy |
| Time Stretch | 0.9 - 1.1 | Scale dt values |

## Data Statistics

After parsing:

| Split | Samples | Usage |
|-------|---------|-------|
| Train | 4,838 | Model training |
| Val | 537 | Hyperparameter tuning |
| Test | 6,804 | Final evaluation |

**Character vocabulary:** 54 unique characters
```
 .ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```

**Label statistics:**
- Min length: 1 character
- Max length: 69 characters
- Average length: 32 characters

## Usage Example

```python
from src.data.preprocessing import StrokePreprocessor
from src.data.dataset import OnlineHTRDataset, collate_fn
from torch.utils.data import DataLoader

# Initialize preprocessor
preprocessor = StrokePreprocessor(
    target_height=60,
    resample_density=20
)

# Load dataset
dataset = OnlineHTRDataset(
    data_path='data/processed/train.json',
    preprocessor=preprocessor,
    augment=True
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# Iterate
for features, labels, feat_lens, label_lens in loader:
    # features: (batch, max_seq_len, 4)
    # labels: (batch, max_label_len)
    # feat_lens: (batch,) actual sequence lengths
    # label_lens: (batch,) actual label lengths
    pass
```

## Inspecting Data

### View raw data structure:
```bash
python scripts/inspect_data.py data/db/lineStrokes/a01/a01-000/a01-000u-01.xml
```

### View processed data:
```bash
python -c "
import json
with open('data/processed/train.json') as f:
    data = json.load(f)
print(f'Samples: {len(data)}')
print(f'First sample: {data[0][\"label\"]}')
print(f'Strokes: {len(data[0][\"strokes\"])}')
"
```

## Troubleshooting

### Missing labels
Some stroke files may not have corresponding labels. The parser skips these automatically.

### Empty strokes
Strokes with < 2 points are passed through without resampling.

### NaN values
The preprocessor replaces NaN/Inf values with 0 after feature extraction.

## References

- IAM-OnDB Paper: Liwicki, M. and Bunke, H. (2005). "IAM-OnDB - an on-line English sentence database acquired from handwritten text on a whiteboard"
- CTC Loss: Graves, A. et al. (2006). "Connectionist Temporal Classification"
