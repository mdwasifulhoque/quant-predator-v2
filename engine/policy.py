"""
engine/policy.py
================
Custom LSTM-based feature extractor for the PPO policy.

Defined in its own module so it can be imported identically in both
trainer.py (during training) and live_trader.py (during inference).
SB3 requires the extractor class to be importable at model-load time.

ARCHITECTURE:
    Input (842,) → reshape (24, 35) → LSTM(128, 2 layers)
    → final hidden (128,) → LayerNorm → concat state (2,) → (130,)
    → passed to MLP policy head [256, 128] → actions / value
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from core.config import ModelConfig


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    LSTM sequence encoder that reads the 24-bar market window.

    Unlike SB3's RecurrentPPO (which maintains hidden state across steps),
    this extractor treats each observation as a self-contained sequence.
    The 24 bars are the full context; the LSTM summarises them into a
    single 128-dim state vector — analogous to reading a sentence with BERT.

    Args:
        observation_space: Passed automatically by SB3.
        lstm_hidden:        Hidden units per LSTM layer.
        n_layers:           Number of stacked LSTM layers.
        dropout:            Dropout between LSTM layers (0 if n_layers=1).
        n_features:         Feature count per time step (must match ModelConfig).
        window_size:        Time steps per observation (must match ModelConfig).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        lstm_hidden:  int = ModelConfig.LSTM_HIDDEN,
        n_layers:     int = ModelConfig.LSTM_LAYERS,
        dropout:      float = ModelConfig.LSTM_DROPOUT,
        n_features:   int = ModelConfig.N_FEATURES,
        window_size:  int = ModelConfig.WINDOW_SIZE,
    ) -> None:
        # features_dim = LSTM output + 2 position/pnl scalars
        super().__init__(observation_space, features_dim=lstm_hidden + 2)

        self.lstm_hidden = lstm_hidden
        self.n_features  = n_features
        self.window_size = window_size

        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = lstm_hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        # LayerNorm stabilises training on normalised financial time-series
        self.layer_norm = nn.LayerNorm(lstm_hidden)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]

        # Split flat observation into market sequence and position state
        market_obs = observations[:, :-2]                                   # (B, 840)
        state_obs  = observations[:, -2:]                                   # (B, 2)

        # Reshape to temporal sequence
        seq = market_obs.reshape(batch, self.window_size, self.n_features)  # (B, 24, 35)

        # LSTM forward — we only need the final hidden state
        _, (hidden, _) = self.lstm(seq)     # hidden: (n_layers, B, 128)
        last_hidden = self.layer_norm(hidden[-1])                           # (B, 128)

        # Append position/pnl scalars so the policy sees account state directly
        return torch.cat([last_hidden, state_obs], dim=1)                   # (B, 130)
