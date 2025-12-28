"""BiLSTM + CTC model for Online HTR."""

import torch
import torch.nn as nn


class OnlineHTRModel(nn.Module):
    """Bidirectional LSTM with CTC for online handwriting recognition."""

    def __init__(self,
                 input_channels=4,
                 hidden_size=256,
                 num_layers=3,
                 num_classes=80,
                 dropout=0.3):
        """
        Args:
            input_channels: Number of input features (dx, dy, dt, pen_state)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (vocab size + 1 for CTC blank)
            dropout: Dropout probability
        """
        super(OnlineHTRModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

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

        # Bidirectional LSTM
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
            x: Input tensor (batch_size, seq_len, input_channels)
            lengths: Actual sequence lengths (batch_size,)
        Returns:
            output: Log probabilities (batch_size, seq_len, num_classes)
        """
        batch_size = x.size(0)

        # Conv expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, channels)

        # Pack padded sequence for LSTM efficiency
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        x, _ = self.lstm(x)

        # Unpack
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Output projection
        x = self.fc(x)

        return x

    def get_output_lengths(self, input_lengths):
        """Get output sequence lengths (same as input for this architecture)."""
        return input_lengths
