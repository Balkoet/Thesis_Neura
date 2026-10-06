#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from sympy import false
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    power_path: str
    weather_path: str
    output_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    val_fraction: float
    patience: int

    embedding_dim: int
    prediction_year: int
    prediction_month: int

HIGH_CONSUM = [ #HIGH CONSUMPTION GROUP OF SUBSTATIONS
    "YS1",
    "YS2",
    "YS3",
    "YS4",
    "YS5",
    "YS_FARM",
]

LOW_CONSUM = [ #LOW CONSUMPTION GROUP OF SUBSTATIONS

    "Elnet_239",
    "Elnet_236",

]

LOWEST_CONSUM = [#LOWEST CONSUMPTION GROUP OF SUBSTATIONS
    "Elnet_227",

]

def resolve_csv_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    if path.suffix.lower() != ".csv":
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError(f"File not found: {path_str}")


def clean_substation_name(raw_name: str) -> str:
    return raw_name.replace("(kW)", "").strip().replace(" ", "_")


def load_weather_data(weather_path: Path) -> pd.DataFrame:
    weather_df = pd.read_csv(weather_path)
    if "date" not in weather_df.columns:
        raise ValueError("Weather data must include a 'date' column")

    weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce").dt.normalize()
    weather_df = weather_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    weather_cols = [c for c in weather_df.columns if c != "date"]
    for col in weather_cols:
        weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
        weather_df[col] = weather_df[col].interpolate(limit_direction="both")
        weather_df[col] = weather_df[col].ffill().bfill()
        weather_df[col] = weather_df[col].fillna(weather_df[col].median())

    return weather_df


def add_time_features(df: pd.DataFrame) -> None:
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)


def load_and_prepare_data(power_path: Path, weather_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    power_df = pd.read_csv(power_path)
    if "Timestamp" not in power_df.columns:
        raise ValueError("Power data must include a 'Timestamp' column")

    power_df["Timestamp"] = pd.to_datetime(power_df["Timestamp"], errors="coerce")
    power_df = power_df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    substation_cols = [c for c in power_df.columns if c != "Timestamp"]
    long_df = power_df.melt(
        id_vars=["Timestamp"],
        value_vars=substation_cols,
        var_name="substation",
        value_name="target_kw",
    )
    long_df["substation"] = long_df["substation"].map(clean_substation_name)
    long_df["target_kw"] = pd.to_numeric(long_df["target_kw"], errors="coerce")

    long_df = long_df.sort_values(["substation", "Timestamp"]).reset_index(drop=True)
    long_df["target_kw"] = (
        long_df.groupby("substation")["target_kw"].transform(lambda s: s.ffill().bfill())
    )
    long_df["target_kw"] = long_df["target_kw"].fillna(long_df["target_kw"].median())

    long_df = long_df.rename(columns={"Timestamp": "timestamp"})
    long_df["date"] = long_df["timestamp"].dt.normalize()

    weather_df = load_weather_data(weather_path)
    df = long_df.merge(weather_df, on="date", how="left")

    add_time_features(df)
    df = add_lag_features(df)

    numeric_feature_cols = [
        "avg_temperature_c",
        "avg_wind_speed_kmh",
        "rain_mm",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",

    ]

    for col in numeric_feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df, numeric_feature_cols


def build_prediction_frame(
    weather_path: Path,
    stations: List[str],
    prediction_year: int,
    prediction_month: int,
) -> pd.DataFrame:
    month_start = pd.Timestamp(year=prediction_year, month=prediction_month, day=1)
    month_end = month_start + pd.offsets.MonthBegin(1)

    weather_df = load_weather_data(weather_path)
    full_dates = pd.date_range(month_start, month_end - pd.Timedelta(days=1), freq="D")
    weather_month = pd.DataFrame({"date": full_dates}).merge(weather_df, on="date", how="left")

    weather_cols = [c for c in weather_month.columns if c != "date"]
    for col in weather_cols:
        weather_month[col] = weather_month[col].interpolate(limit_direction="both").ffill().bfill()

    hourly_timestamps = pd.date_range(month_start, month_end - pd.Timedelta(hours=1), freq="h")
    ts_df = pd.DataFrame({"timestamp": hourly_timestamps})
    stations_df = pd.DataFrame({"substation": stations})

    forecast_df = ts_df.merge(stations_df, how="cross")
    forecast_df["date"] = forecast_df["timestamp"].dt.normalize()
    forecast_df = forecast_df.merge(weather_month, on="date", how="left")
    add_time_features(forecast_df)
    forecast_df["target_kw"] = 0.0

    return forecast_df


def add_split_and_encodings(df: pd.DataFrame, val_fraction: float):
    rng = np.random.default_rng()

    # unique days
    unique_days = df["date"].drop_duplicates().to_numpy()
    rng.shuffle(unique_days)

    n_val_days = max(1, int(round(len(unique_days) * val_fraction)))
    val_days = set(unique_days[:n_val_days])

    # Split by whole day
    df["split"] = np.where(df["date"].isin(val_days), "val", "train")

    # Station encoding
    stations = sorted(df["substation"].unique())
    station_to_idx = {s: i for i, s in enumerate(stations)}
    df["substation_idx"] = df["substation"].map(station_to_idx).astype(int)


    return df, station_to_idx

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["substation", "timestamp"]).copy()
    g = df.groupby("substation")["target_kw"]

    df["lag_1h"] = g.shift(1)
    df["lag_24h"] = g.shift(24)
    df["lag_168h"] = g.shift(168)  # same hour last week
    df["lag_672h"] = g.shift(672)  # month lag

    # optional rolling stats from past values only
    df["roll_mean_24h"] = g.shift(1).rolling(24).mean().reset_index(level=0, drop=True)
    df["roll_std_24h"] = g.shift(1).rolling(24).std().reset_index(level=0, drop=True)

    return df

class PowerDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_cols: List[str]):
        self.x = frame[feature_cols].to_numpy(dtype=np.float32, copy=True)
        self.station_idx = frame["substation_idx"].to_numpy(dtype=np.int64, copy=True)
        self.y = frame["target_kw"].to_numpy(dtype=np.float32, copy=True)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.station_idx[idx], self.y[idx]


class FFNNRegressor(nn.Module):
    def __init__(self, num_numeric_features: int, num_stations: int, embedding_dim: int, hidden_sizes=(64, 32, 16),):
        super().__init__()
        self.embedding = nn.Embedding(num_stations, embedding_dim)
        input_dim = num_numeric_features + embedding_dim

        layers = []
        prev = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

            prev = h

        layers.append(nn.Linear(prev, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, station_idx: torch.Tensor) -> torch.Tensor:
        station_emb = self.embedding(station_idx)
        x = torch.cat([x_num, station_emb], dim=1)
        return self.network(x).squeeze(1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    nonzero_mask = np.abs(y_true) > 1e-6
    if np.any(nonzero_mask):
        denom = np.abs(y_true[nonzero_mask])
        mape = float(np.mean(np.abs(errors[nonzero_mask]) / denom) * 100)
    else:
        mape = float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for x_num, station_idx, y in loader:
            x_num = x_num.to(device)
            station_idx = station_idx.to(device)
            y_hat = model(x_num, station_idx).cpu().numpy()
            preds.append(y_hat)
            targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(preds)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    model_path: Path,
    use_smooth_l1: bool,
) -> None:
    criterion = (
        nn.SmoothL1Loss()
        if use_smooth_l1
        else nn.MSELoss()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_rmse = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x_num, station_idx, y in train_loader:
            x_num = x_num.to(device)
            station_idx = station_idx.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x_num, station_idx), y)
            loss.backward()
            optimizer.step()

        val_y, val_pred = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_y, val_pred)

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break


def run_pipeline( cfg: Config,stations_to_use, normalize_target=False, use_smooth_l1=False,  hidden_sizes=(64, 32),):


    power_path = resolve_csv_path(cfg.power_path)
    weather_path = resolve_csv_path(cfg.weather_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, numeric_feature_cols = load_and_prepare_data(power_path, weather_path)
    df = df[df["substation"].isin(stations_to_use)].copy()

    month_start = pd.Timestamp(
        year=2025,
        month=2,
        day=1,
    )
    month_end = month_start + pd.offsets.MonthBegin(1)


    held_out_month = df[
        (df["timestamp"] >= month_start) &
        (df["timestamp"] < month_end)
        ].copy()

    # REMOVE PREDICTION MONTH
    df = df[
        (df["timestamp"] < month_start) |
        (df["timestamp"] >= month_end)
        ].reset_index(drop=True)

    target_scalers = {}

    if normalize_target: #normalization

        for station in df["substation"].unique():

            mask = df["substation"] == station

            mean = df.loc[mask, "target_kw"].mean()
            std = df.loc[mask, "target_kw"].std()

            if std < 1e-6:
                std = 1

            target_scalers[station] = (mean, std)

            df.loc[mask, "target_kw"] = (
                                                df.loc[mask, "target_kw"] - mean
                                        ) / std

    df, station_to_idx = add_split_and_encodings(
        df,
        cfg.val_fraction,

    )

    train_mask = df["split"] == "train"

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(df.loc[train_mask, numeric_feature_cols])
    df.loc[:, numeric_feature_cols] = imputer.transform(df[numeric_feature_cols])
    df.loc[train_mask, numeric_feature_cols] = scaler.fit_transform(df.loc[train_mask, numeric_feature_cols])
    df.loc[~train_mask, numeric_feature_cols] = scaler.transform(df.loc[~train_mask, numeric_feature_cols])

    train_df = df[train_mask].copy()
    val_df = df[~train_mask].copy()

    train_dataset = PowerDataset(train_df, numeric_feature_cols)
    val_dataset = PowerDataset(val_df, numeric_feature_cols)
    full_dataset = PowerDataset(df, numeric_feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    embedding_dim = cfg.embedding_dim if cfg.embedding_dim > 0 else min(16, max(2, int(math.sqrt(len(station_to_idx)))))
    model = FFNNRegressor(
        num_numeric_features=len(numeric_feature_cols),
        num_stations=len(station_to_idx),
        embedding_dim=embedding_dim,
        hidden_sizes=hidden_sizes,
    ).to(device)

    best_model_path = output_dir / "best_model.pt"
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        patience=cfg.patience,
        device=device,
        model_path=best_model_path,
        use_smooth_l1=use_smooth_l1,
    )

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    all_y, all_pred = evaluate(model, full_loader, device)
    val_y, val_pred = evaluate(model, val_loader, device)

    df["actual_kw"] = all_y
    df["predicted_kw"] = all_pred

    overall_metrics = compute_metrics(val_y, val_pred)
    per_substation_metrics = []
    for station, group in df[df["split"] == "val"].groupby("substation"):
        metrics = compute_metrics(group["actual_kw"].to_numpy(), group["predicted_kw"].to_numpy())
        metrics["substation"] = station
        per_substation_metrics.append(metrics)

    metrics_path = output_dir / "metrics_overall.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    per_station_df = pd.DataFrame(per_substation_metrics).sort_values("substation")
    per_station_df.to_csv(output_dir / "metrics_per_substation.csv", index=False)

    stations = sorted(station_to_idx.keys())
    forecast_df = build_prediction_frame(
        weather_path=weather_path,
        stations=stations,
        prediction_year=cfg.prediction_year,
        prediction_month=cfg.prediction_month,
    )
    forecast_df["substation_idx"] = forecast_df["substation"].map(station_to_idx).astype(int)
    forecast_df.loc[:, numeric_feature_cols] = imputer.transform(forecast_df[numeric_feature_cols])
    forecast_df.loc[:, numeric_feature_cols] = scaler.transform(forecast_df[numeric_feature_cols])

    forecast_dataset = PowerDataset(forecast_df, numeric_feature_cols)
    forecast_loader = DataLoader(forecast_dataset, batch_size=cfg.batch_size, shuffle=False)
    _, forecast_pred = evaluate(model, forecast_loader, device)

    forecast_df["predicted_kw"] = forecast_pred

    if normalize_target: #normalise, only for low consumption

        for station in forecast_df["substation"].unique():
            mean, std = target_scalers[station]

            mask = forecast_df["substation"] == station

            forecast_df.loc[mask, "predicted_kw"] = (
                    forecast_df.loc[mask, "predicted_kw"] * std
                    + mean
            )

    month_start = pd.Timestamp(year=cfg.prediction_year, month=cfg.prediction_month, day=1)
    month_end = month_start + pd.offsets.MonthBegin(1)

    month_actuals = held_out_month[
        ["timestamp", "substation", "target_kw"]
    ].rename(columns={"target_kw": "actual_kw"})

    predictions_long = forecast_df[["timestamp", "substation", "predicted_kw"]].merge(
        month_actuals,
        on=["timestamp", "substation"],
        how="left",
    )

    predictions_wide = predictions_long.pivot(
        index="timestamp",
        columns="substation",
        values="predicted_kw",
    ).reset_index()

    predictions_wide = predictions_wide.rename(
        columns={"timestamp": "Timestamp", **{s: f"{s}(kW)" for s in station_to_idx.keys()}}
    )

    predictions_wide["Timestamp"] = pd.to_datetime(predictions_wide["Timestamp"]).dt.strftime("%m/%d/%Y %H:%M")

    ordered_cols = ["Timestamp"] + [f"{s}(kW)" for s in sorted(station_to_idx.keys())]
    predictions_wide = predictions_wide.reindex(columns=ordered_cols)

    predictions_wide.to_csv(output_dir / "predictions.csv", index=False)

    training_metadata = {
        "power_path": str(power_path),
        "weather_path": str(weather_path),
        "numeric_feature_cols": numeric_feature_cols,
        "station_to_idx": station_to_idx,
        "embedding_dim": embedding_dim,
        "val_fraction": cfg.val_fraction,

    }
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2)

    dump(imputer, output_dir / "imputer.joblib")
    dump(scaler, output_dir / "scaler.joblib")

    print("Training complete")
    print(f"Validation metrics: {overall_metrics}")
    print(f"Saved artifacts to: {output_dir.resolve()}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a feed-forward neural network with Adam to forecast hourly substation power consumption."
    )
    parser.add_argument("--power-path", default="data_power_final", help="Path to power data file (with or without .csv)")
    parser.add_argument("--weather-path", default="weather_final", help="Path to weather data file (with or without .csv)")
    parser.add_argument("--output-dir", default="outputs", help="Directory to store model outputs and predictions")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.4) #Validation %
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=0,
        help="Substation embedding dimension. Use 0 for automatic size.",
    )
    parser.add_argument("--prediction-year", type=int, default=2024, help="Year to export predictions for") #Prediction year
    parser.add_argument("--prediction-month", type=int, default=10, help="Month (1-12) to export predictions for") #Prediction month

    args = parser.parse_args()
    args = parser.parse_args()
    if not (0 < args.val_fraction < 1):
        raise ValueError("--val-fraction must be between 0 and 1")
    if not (1 <= args.prediction_month <= 12):
        raise ValueError("--prediction-month must be between 1 and 12")

    return Config(
        power_path=args.power_path,
        weather_path=args.weather_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        patience=args.patience,

        embedding_dim=args.embedding_dim,
        prediction_year=args.prediction_year,
        prediction_month=args.prediction_month,
    )


if __name__ == "__main__":

    cfg = parse_args()

    for i in range(10):
        cfg.output_dir = f"outputs_high_run_{i}"

        run_pipeline(
            cfg,
            HIGH_CONSUM,
            normalize_target=False,
            use_smooth_l1=False,
            hidden_sizes=(64, 32),
        )


        cfg.output_dir = f"outputs_low_run_{i}"

        run_pipeline(
            cfg,
            LOW_CONSUM,
            normalize_target=False,
            use_smooth_l1=False,
            hidden_sizes=(32, 16),
        )

        cfg.output_dir = f"outputs_lowest_run_{i}"

        run_pipeline(
            cfg,
            LOWEST_CONSUM,
            normalize_target=True,
            use_smooth_l1=True,
            hidden_sizes=(32, 16),

        )