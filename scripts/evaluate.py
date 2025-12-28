#!/usr/bin/env python3
"""Evaluation script for Online HTR model."""

import os
import sys
import argparse
import yaml
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.model import OnlineHTRModel
from src.data.dataset import OnlineHTRDataset, collate_fn
from src.data.preprocessing import StrokePreprocessor
from src.training.metrics import calculate_cer, calculate_wer, ctc_greedy_decode


def evaluate_model(config_path, checkpoint_path, test_data_path, output_path=None):
    """Evaluate model on test data."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
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
        augment=False,
        char_to_idx=checkpoint['char_to_idx']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    num_classes = len(checkpoint['char_to_idx']) + 1
    model = OnlineHTRModel(
        input_channels=config['model']['input_channels'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=num_classes,
        dropout=0.0  # No dropout for inference
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint CER: {checkpoint.get('cer', 'N/A'):.2f}%")

    # Evaluate
    all_preds = []
    all_targets = []
    results = []

    print("\nEvaluating...")
    with torch.no_grad():
        for features, labels, feat_lens, label_lens in tqdm(test_loader):
            if feat_lens.min() == 0:
                continue

            features = features.to(device)

            outputs = model(features, feat_lens)
            outputs_np = outputs.cpu().numpy()

            for i in range(outputs_np.shape[0]):
                seq_len = feat_lens[i].item()
                pred = outputs_np[i, :seq_len, :].argmax(axis=1)
                pred = ctc_greedy_decode(pred)
                pred_text = test_dataset.decode_label(pred)

                target = labels[i, :label_lens[i]].numpy().tolist()
                target_text = test_dataset.decode_label(target)

                all_preds.append(pred_text)
                all_targets.append(target_text)
                results.append({
                    'target': target_text,
                    'prediction': pred_text,
                    'correct': pred_text == target_text
                })

    # Calculate metrics
    cer = calculate_cer(all_preds, all_targets)
    wer = calculate_wer(all_preds, all_targets)

    # Count exact matches
    exact_matches = sum(1 for r in results if r['correct'])
    accuracy = 100 * exact_matches / len(results) if results else 0

    print(f"\n{'='*50}")
    print("Test Results:")
    print(f"{'='*50}")
    print(f"  Samples:        {len(results)}")
    print(f"  CER:            {cer:.2f}%")
    print(f"  WER:            {wer:.2f}%")
    print(f"  Exact Match:    {accuracy:.2f}% ({exact_matches}/{len(results)})")
    print(f"{'='*50}")

    # Print sample predictions
    print("\nSample predictions:")
    for i in range(min(10, len(results))):
        status = "✓" if results[i]['correct'] else "✗"
        print(f"  [{status}] Target: '{results[i]['target']}'")
        print(f"       Pred:   '{results[i]['prediction']}'")
        print()

    # Save results
    if output_path:
        output = {
            'metrics': {
                'cer': cer,
                'wer': wer,
                'accuracy': accuracy,
                'num_samples': len(results)
            },
            'results': results
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")

    return cer, wer, accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate Online HTR model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/db',
                        help='Path to test data')
    parser.add_argument('--output', type=str, default='test_results.json',
                        help='Path to save results')
    args = parser.parse_args()

    evaluate_model(args.config, args.checkpoint, args.data, args.output)


if __name__ == "__main__":
    main()
