#!/usr/bin/env python3

from __future__ import annotations

import math

import torch
from torch import nn

from cnn_1d_forecast_february_visible import (
    FEATURE_COLUMNS,
    ExperimentConfig,
    parse_args,
    run_experiment,
)


class CNNLSTM(nn.Module):
    def __init__(self, feature_count: int, station_count: int):
        super().__init__()
        embedding_dim = max(2, min(8, int(math.ceil(math.sqrt(station_count)))))
        self.convolution = nn.Sequential(
            nn.Conv1d(feature_count, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(64 + embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, sequence: torch.Tensor, station_index: torch.Tensor) -> torch.Tensor:
        convolved = self.convolution(sequence).transpose(1, 2)
        recurrent, _ = self.lstm(convolved)
        temporal_state = recurrent[:, -1, :]
        station = self.station_embedding(station_index)
        return self.head(torch.cat([temporal_state, station], dim=1)).squeeze(1)


def cnn_lstm_factory(feature_count: int, station_count: int) -> nn.Module:
    return CNNLSTM(feature_count, station_count)


if __name__ == "__main__":
    config: ExperimentConfig = parse_args(
        default_output="outputs_cnn_lstm_february_visible"
    )

    config.prediction_year = 2024 #Prediction Year
    config.prediction_month = 10 #Prediction Month
    config.val_fraction = 0.4 #Validation %

    run_experiment(
        config,
        factory=cnn_lstm_factory,
        architecture_label="CNN-LSTM hybrid (February visible)",
    )
