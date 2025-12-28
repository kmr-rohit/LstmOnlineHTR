#!/usr/bin/env python3
"""Inference script for Online HTR model."""

import os
import sys
import argparse
import yaml
import json
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.model import OnlineHTRModel
from src.data.preprocessing import StrokePreprocessor
from src.training.metrics import ctc_greedy_decode


class HTRPredictor:
    """Predictor class for online handwriting recognition."""

    def __init__(self, config_path, checkpoint_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Initialize preprocessor
        self.preprocessor = StrokePreprocessor(
            target_height=self.config['preprocessing']['target_height'],
            resample_density=self.config['preprocessing']['resample_density']
        )

        # Character mapping
        self.char_to_idx = self.checkpoint['char_to_idx']
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        # Initialize model
        num_classes = len(self.char_to_idx) + 1
        self.model = OnlineHTRModel(
            input_channels=self.config['model']['input_channels'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            num_classes=num_classes,
            dropout=0.0
        ).to(self.device)

        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, strokes):
        """
        Recognize text from stroke data.

        Args:
            strokes: List of stroke dictionaries with 'points' containing x, y, t, pen_down

        Returns:
            text: Recognized text
            confidence: Average confidence score
        """
        # Preprocess
        features = self.preprocessor.process(strokes)
        features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        feat_len = torch.LongTensor([features.size(1)])

        # Inference
        with torch.no_grad():
            outputs = self.model(features, feat_len)
            probs = torch.softmax(outputs, dim=2)

            # Greedy decode
            pred = outputs[0].argmax(dim=1).cpu().numpy()
            pred = ctc_greedy_decode(pred)

            # Decode to text
            text = ''.join([self.idx_to_char.get(i, '') for i in pred])

            # Confidence: average max probability
            confidence = probs[0].max(dim=1)[0].mean().item()

        return text, confidence

    def predict_from_file(self, json_path):
        """Recognize text from a JSON file containing stroke data."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        strokes = data.get('strokes', data.get('stroke_data', []))
        return self.predict(strokes)


def main():
    parser = argparse.ArgumentParser(description='Run inference with Online HTR model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file with stroke data')
    args = parser.parse_args()

    predictor = HTRPredictor(args.config, args.checkpoint)

    text, confidence = predictor.predict_from_file(args.input)
    print(f"Recognized text: {text}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
