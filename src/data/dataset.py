"""Dataset classes for Online HTR."""

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

from .preprocessing import StrokePreprocessor
from .augmentation import StrokeAugmentation


class OnlineHTRDataset(Dataset):
    """Dataset for online handwriting recognition."""

    def __init__(self, data_path, preprocessor=None, augment=False, augmentor=None, char_to_idx=None):
        """
        Args:
            data_path: Path to JSON data file or directory
            preprocessor: StrokePreprocessor instance
            augment: Whether to apply augmentation
            augmentor: StrokeAugmentation instance
            char_to_idx: Pre-built character mapping (for val/test sets)
        """
        self.data = self.load_data(data_path)
        self.preprocessor = preprocessor or StrokePreprocessor()
        self.augment = augment
        self.augmentor = augmentor or StrokeAugmentation()

        # Build or use provided character mapping
        if char_to_idx is not None:
            self.char_to_idx = char_to_idx
            self.idx_to_char = {i: c for c, i in char_to_idx.items()}
            self.chars = sorted(char_to_idx.keys())
        else:
            self.chars = self.build_char_set()
            self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}  # 0 reserved for CTC blank
            self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

    def load_data(self, data_path):
        """Load dataset from JSON file or directory."""
        if os.path.isfile(data_path):
            with open(data_path, 'r') as f:
                return json.load(f)
        elif os.path.isdir(data_path):
            # Load all JSON files from directory
            data = []
            for filename in os.listdir(data_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(data_path, filename)
                    with open(filepath, 'r') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            data.extend(file_data)
                        else:
                            data.append(file_data)
            return data
        else:
            raise ValueError(f"Data path not found: {data_path}")

    def build_char_set(self):
        """Build character vocabulary from data."""
        chars = set()
        for sample in self.data:
            label = sample.get('label', sample.get('text', ''))
            chars.update(label)
        return sorted(list(chars))

    def encode_label(self, text):
        """Encode text to indices."""
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode_label(self, indices):
        """Decode indices to text."""
        return ''.join([self.idx_to_char.get(i, '') for i in indices if i != 0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Get strokes - handle different formats
        strokes = sample.get('strokes', sample.get('stroke_data', []))

        # Preprocess
        features = self.preprocessor.process(strokes)

        # Augment if training
        if self.augment:
            features = self.augmentor.augment(features)

        # Get label
        label_text = sample.get('label', sample.get('text', ''))
        label = self.encode_label(label_text)

        # Convert to tensors
        features = torch.FloatTensor(features)
        label = torch.LongTensor(label) if label else torch.LongTensor([0])

        return features, label, len(features), len(label)


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    features, labels, feat_lens, label_lens = zip(*batch)

    # Pad sequences
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    feat_lens = torch.LongTensor(feat_lens)
    label_lens = torch.LongTensor(label_lens)

    return features_padded, labels_padded, feat_lens, label_lens
