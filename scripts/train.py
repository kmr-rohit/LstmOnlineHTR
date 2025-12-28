#!/usr/bin/env python3
"""Training script for Online HTR model."""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.model import OnlineHTRModel
from src.data.dataset import OnlineHTRDataset, collate_fn
from src.data.preprocessing import StrokePreprocessor
from src.data.augmentation import StrokeAugmentation
from src.training.metrics import calculate_cer, calculate_wer, ctc_greedy_decode


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
        self.augmentor = StrokeAugmentation(
            scale_range=tuple(self.config['augmentation']['scale_range']),
            rotation_range=tuple(self.config['augmentation']['rotation_range']),
            noise_std=self.config['augmentation']['noise_std'],
            time_stretch_range=tuple(self.config['augmentation']['time_stretch_range'])
        )

        # Load and split data
        self.setup_data()

        # Initialize model
        num_classes = len(self.train_dataset.chars) + 1  # +1 for CTC blank
        self.model = OnlineHTRModel(
            input_channels=self.config['model']['input_channels'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            num_classes=num_classes,
            dropout=self.config['model']['dropout']
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

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
        self.best_cer = float('inf')
        self.patience_counter = 0

        # Create checkpoint directory
        os.makedirs(self.config['paths']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['log_dir'], exist_ok=True)

    def setup_data(self):
        """Load data and create train/val split."""
        data_dir = self.config['paths']['data_dir']

        # Check if separate train/val files exist
        train_path = os.path.join(data_dir, 'train.json')
        val_path = os.path.join(data_dir, 'val.json')

        if os.path.exists(train_path) and os.path.exists(val_path):
            # Use separate files
            self.train_dataset = OnlineHTRDataset(
                data_path=train_path,
                preprocessor=self.preprocessor,
                augment=self.config['augmentation']['enabled'],
                augmentor=self.augmentor
            )
            self.val_dataset = OnlineHTRDataset(
                data_path=val_path,
                preprocessor=self.preprocessor,
                augment=False,
                char_to_idx=self.train_dataset.char_to_idx
            )
        else:
            # Load all data and split
            full_dataset = OnlineHTRDataset(
                data_path=data_dir,
                preprocessor=self.preprocessor,
                augment=False
            )

            # Split 90/10
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size

            train_data, val_data = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Wrap with augmentation for training
            self.train_dataset = full_dataset
            self.train_dataset.augment = self.config['augmentation']['enabled']
            self.train_dataset.augmentor = self.augmentor

            # Create subset wrappers
            self.train_indices = train_data.indices
            self.val_indices = val_data.indices

            print(f"Dataset split: {len(self.train_indices)} train, {len(self.val_indices)} val")

        # Create dataloaders
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training'].get('num_workers', 0)

        if hasattr(self, 'train_indices'):
            # Use subset samplers
            from torch.utils.data import SubsetRandomSampler
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(self.train_indices),
                collate_fn=collate_fn,
                num_workers=num_workers
            )
            # Disable augmentation for validation
            self.train_dataset.augment = False
            self.val_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(self.val_indices),
                collate_fn=collate_fn,
                num_workers=num_workers
            )
            # Re-enable for training
            self.train_dataset.augment = self.config['augmentation']['enabled']
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers
            )

        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Vocabulary size: {len(self.train_dataset.chars)} characters")
        print(f"Characters: {''.join(self.train_dataset.chars[:50])}{'...' if len(self.train_dataset.chars) > 50 else ''}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for features, labels, feat_lens, label_lens in pbar:
            # Skip empty batches
            if feat_lens.min() == 0 or label_lens.min() == 0:
                continue

            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(features, feat_lens)

            # CTC loss expects (T, N, C)
            outputs = outputs.log_softmax(2)
            outputs = outputs.transpose(0, 1)

            # Calculate loss
            loss = self.criterion(outputs, labels, feat_lens, label_lens)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

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
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_targets = []

        # Temporarily disable augmentation
        if hasattr(self, 'train_indices'):
            self.train_dataset.augment = False

        with torch.no_grad():
            for features, labels, feat_lens, label_lens in tqdm(self.val_loader, desc="Validating"):
                if feat_lens.min() == 0 or label_lens.min() == 0:
                    continue

                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(features, feat_lens)
                outputs_log = outputs.log_softmax(2)
                outputs_t = outputs_log.transpose(0, 1)

                # Calculate loss
                loss = self.criterion(outputs_t, labels, feat_lens, label_lens)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1

                # Decode predictions
                outputs_np = outputs.cpu().numpy()
                for i in range(outputs_np.shape[0]):
                    seq_len = feat_lens[i].item()
                    pred = outputs_np[i, :seq_len, :].argmax(axis=1)
                    pred = ctc_greedy_decode(pred)
                    pred_text = self.train_dataset.decode_label(pred)

                    target = labels[i, :label_lens[i]].cpu().numpy().tolist()
                    target_text = self.train_dataset.decode_label(target)

                    all_preds.append(pred_text)
                    all_targets.append(target_text)

        # Re-enable augmentation
        if hasattr(self, 'train_indices'):
            self.train_dataset.augment = self.config['augmentation']['enabled']

        # Calculate metrics
        avg_loss = total_loss / max(num_batches, 1)
        cer = calculate_cer(all_preds, all_targets)
        wer = calculate_wer(all_preds, all_targets)

        # Print some examples
        print("\n  Sample predictions:")
        for i in range(min(3, len(all_preds))):
            print(f"    Target: '{all_targets[i]}'")
            print(f"    Pred:   '{all_preds[i]}'")
            print()

        return avg_loss, cer, wer

    def save_checkpoint(self, epoch, val_loss, cer, wer, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'cer': cer,
            'wer': wer,
            'config': self.config,
            'char_to_idx': self.train_dataset.char_to_idx,
            'chars': self.train_dataset.chars
        }

        # Save latest
        torch.save(checkpoint, os.path.join(
            self.config['paths']['checkpoint_dir'], 'latest_model.pth'
        ))

        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(
                self.config['paths']['checkpoint_dir'], 'best_model.pth'
            ))

    def train(self):
        print("Starting training...")
        print(f"Config: {self.config['model']['name']}")
        print("-" * 50)

        history = []

        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, cer, wer = self.validate()

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Logging
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  CER:        {cer:.2f}%")
            print(f"  WER:        {wer:.2f}%")
            print(f"  LR:         {current_lr:.6f}")

            # Track history
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'cer': cer,
                'wer': wer,
                'lr': current_lr
            })

            # Save best model
            is_best = cer < self.best_cer
            if is_best:
                self.best_cer = cer
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print("  -> New best model!")
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_loss, cer, wer, is_best)

            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping']['patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

            print("-" * 50)

        # Save training history
        with open(os.path.join(self.config['paths']['log_dir'], 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        print("\nTraining complete!")
        print(f"Best CER: {self.best_cer:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train Online HTR model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    trainer = Trainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
