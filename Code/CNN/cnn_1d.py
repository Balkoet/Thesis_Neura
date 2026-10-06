
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


TARGET_COLUMNS = [
    "YS1(kW)", "YS2(kW)", "YS3(kW)", "YS4(kW)", "YS5(kW)",
    "YS_FARM(kW)", "Elnet_227(kW)", "Elnet_239(kW)", "Elnet_236(kW)",
]
GROUPS: Mapping[str, List[str]] = {
    "high": ["YS1(kW)", "YS2(kW)", "YS3(kW)", "YS4(kW)", "YS5(kW)", "YS_FARM(kW)"],
    "low": ["Elnet_239(kW)", "Elnet_236(kW)"],
    "lowest": ["Elnet_227(kW)"],
}
FEATURE_COLUMNS = [
    "avg_temperature_c", "avg_wind_speed_kmh", "rain_mm",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend",
]
WEATHER_COLUMNS = ["avg_temperature_c", "avg_wind_speed_kmh", "rain_mm"]
DEFAULT_SEEDS = tuple(range(42, 52))


@dataclass
class ExperimentConfig:
    power_path: str
    weather_path: str
    output_dir: str
    prediction_year: int = 2025
    prediction_month: int = 2
    lookback: int = 168
    val_fraction: float = 0.10
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20
    min_delta: float = 1e-5
    num_workers: int = 0

    def prediction_bounds(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        start = pd.Timestamp(year=self.prediction_year, month=self.prediction_month, day=1)
        return start, start + pd.offsets.MonthBegin(1)

    def month_tag(self) -> str:
        return f"{self.prediction_year:04d}_{self.prediction_month:02d}"


@dataclass(frozen=True)
class Sample:
    end_position: int
    station_index: int
    target: float
    timestamp: pd.Timestamp


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_input_path(path_text: str) -> Path:
    requested = Path(path_text)
    names = [requested]
    if requested.suffix.lower() != ".csv":
        names.append(requested.with_suffix(".csv"))
    script_dir = Path(__file__).resolve().parent
    roots = [Path.cwd(), script_dir, script_dir / ".venv" / "Scripts"]
    candidates: List[Path] = []
    for name in names:
        if name.is_absolute():
            candidates.append(name)
        else:
            candidates.extend(root / name for root in roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {path_text}. Checked:\n  {checked}")


def load_power(
    path: Path,
    prediction_start: pd.Timestamp | None = None,
    prediction_end: pd.Timestamp | None = None,
    lookback: int = 168,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in ["Timestamp", *TARGET_COLUMNS] if c not in frame.columns]
    if missing:
        raise ValueError(f"Power file is missing columns: {missing}")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    frame = frame.dropna(subset=["Timestamp"]).copy()
    # The source contains duplicate timestamps and one off-grid first reading.
    # Round to its intended hourly resolution, then keep the first occurrence,
    # matching the existing FFNN's duplicate policy.
    frame["Timestamp"] = frame["Timestamp"].dt.round("h")
    frame = frame.sort_values("Timestamp").drop_duplicates("Timestamp", keep="first")
    for column in TARGET_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.set_index("Timestamp").sort_index()
    index_start = frame.index.min()
    index_end = frame.index.max()
    if prediction_start is not None and prediction_end is not None:

        index_start = min(index_start, prediction_start - pd.Timedelta(hours=lookback - 1))
        index_end = max(index_end, prediction_end - pd.Timedelta(hours=1))
    hourly_index = pd.date_range(index_start, index_end, freq="h")
    return frame[TARGET_COLUMNS].reindex(hourly_index).rename_axis("timestamp")


def season_number(month: pd.Series) -> pd.Series:
    return ((month % 12) // 3).astype(int)


def load_weather(
    path: Path,
    hourly_index: pd.DatetimeIndex,
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    weather = pd.read_csv(path)
    missing = [c for c in ["date", *WEATHER_COLUMNS] if c not in weather.columns]
    if missing:
        raise ValueError(f"Weather file is missing columns: {missing}")
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()
    weather = weather.dropna(subset=["date"]).drop_duplicates("date", keep="first")
    for column in WEATHER_COLUMNS:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")


    climatology_source = weather.copy()
    climatology_source["season"] = season_number(climatology_source["date"].dt.month)
    seasonal = climatology_source.groupby("season")[WEATHER_COLUMNS].mean()
    global_means = climatology_source[WEATHER_COLUMNS].mean()
    selected_season = int((prediction_start.month % 12) // 3)
    seasonal_values = {
        column: float(seasonal.loc[selected_season, column]) if selected_season in seasonal.index else float(global_means[column])
        for column in WEATHER_COLUMNS
    }

    all_dates = pd.DataFrame({"date": pd.date_range(hourly_index.min().normalize(), hourly_index.max().normalize(), freq="D")})
    daily = all_dates.merge(weather, on="date", how="left")
    for column in WEATHER_COLUMNS:
        daily[column] = daily[column].interpolate(limit_direction="both")
        daily[column] = daily[column].fillna(global_means[column])

    hourly = pd.DataFrame({"timestamp": hourly_index})
    hourly["date"] = hourly["timestamp"].dt.normalize()
    hourly = hourly.merge(daily, on="date", how="left").set_index("timestamp")
    return hourly[WEATHER_COLUMNS], seasonal_values


def build_covariates(
    power: pd.DataFrame,
    weather_path: Path,
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    weather, seasonal_values = load_weather(weather_path, power.index, prediction_start, prediction_end)
    features = weather.copy()
    hour = features.index.hour
    dow = features.index.dayofweek
    month = features.index.month
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    features["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    features["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    features["is_weekend"] = (dow >= 5).astype(float)
    return features[FEATURE_COLUMNS].astype(np.float32), seasonal_values


def eligible_pool_days(
    power: pd.DataFrame,
    lookback: int,
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
) -> np.ndarray:
    positions = np.arange(len(power))
    has_any_target = power.notna().any(axis=1).to_numpy()
    eligible = (positions >= lookback - 1) & has_any_target
    in_test = (power.index >= prediction_start) & (power.index < prediction_end)
    return np.unique(power.index[eligible & ~in_test].normalize().to_numpy())


def choose_validation_days(pool_days: np.ndarray, fraction: float, seed: int) -> set:
    rng = np.random.default_rng(seed)
    shuffled = pool_days.copy()
    rng.shuffle(shuffled)
    count = max(1, int(round(len(shuffled) * fraction)))
    return set(pd.Timestamp(day) for day in shuffled[:count])


def build_samples(
    power: pd.DataFrame,
    stations: Sequence[str],
    lookback: int,
    validation_days: set,
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
) -> Dict[str, List[Sample]]:
    samples: Dict[str, List[Sample]] = {"train": [], "validation": [], "test": []}
    for position in range(lookback - 1, len(power)):
        timestamp = power.index[position]
        if prediction_start <= timestamp < prediction_end:

            splits = ("train", "test")
        elif timestamp.normalize() in validation_days:
            splits = ("validation",)
        else:
            splits = ("train",)
        for station_index, station in enumerate(stations):
            target = power.iloc[position][station]

            numeric_target = float(target) if pd.notna(target) else math.nan
            for split in splits:
                if split == "test" or pd.notna(target):
                    samples[split].append(Sample(position, station_index, numeric_target, timestamp))
    return samples


class SequenceDataset(Dataset):
    def __init__(
        self,
        feature_values: np.ndarray,
        samples: Sequence[Sample],
        lookback: int,
        target_stats: Mapping[int, Tuple[float, float]] | None = None,
    ):
        self.features = torch.as_tensor(feature_values, dtype=torch.float32)
        self.samples = list(samples)
        self.lookback = lookback
        self.target_stats = target_stats or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        start = sample.end_position - self.lookback + 1
        # Conv1d expects (channels, sequence_length).
        sequence = self.features[start:sample.end_position + 1].transpose(0, 1)
        target = sample.target
        if sample.station_index in self.target_stats:
            mean, std = self.target_stats[sample.station_index]
            target = (target - mean) / std
        return sequence, sample.station_index, np.float32(target)


class SimpleCNN1D(nn.Module):
    def __init__(self, feature_count: int, station_count: int):
        super().__init__()
        embedding_dim = max(2, min(8, int(math.ceil(math.sqrt(station_count)))))
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_count, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.station_embedding = nn.Embedding(station_count, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(64 * 8 + feature_count + embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, sequence: torch.Tensor, station_index: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(sequence).flatten(1)
        current_covariates = sequence[:, :, -1]
        station = self.station_embedding(station_index)
        return self.head(torch.cat([encoded, current_covariates, station], dim=1)).squeeze(1)


def group_training_policy(group_name: str) -> Tuple[bool, Callable[[], nn.Module]]:
    if group_name == "lowest":
        return True, nn.SmoothL1Loss
    return False, nn.MSELoss


def target_statistics(samples: Sequence[Sample], normalize: bool) -> Dict[int, Tuple[float, float]]:
    if not normalize:
        return {}
    grouped: Dict[int, List[float]] = {}
    for sample in samples:
        grouped.setdefault(sample.station_index, []).append(sample.target)
    result = {}
    for station_index, values in grouped.items():
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        result[station_index] = (mean, std if std > 1e-6 else 1.0)
    return result


def make_loader(dataset: Dataset, config: ExperimentConfig, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ExperimentConfig,
    criterion_factory: Callable[[], nn.Module],
    device: torch.device,
) -> Tuple[nn.Module, pd.DataFrame]:
    criterion = criterion_factory()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, config.patience // 4)
    )
    best_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for sequence, station_index, target in train_loader:
            sequence = sequence.to(device, non_blocking=True)
            station_index = station_index.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(sequence, station_index)
            loss = criterion(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_sum += float(loss.item()) * len(target)
            train_count += len(target)

        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for sequence, station_index, target in validation_loader:
                sequence = sequence.to(device, non_blocking=True)
                station_index = station_index.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                loss = criterion(model(sequence, station_index), target)
                validation_sum += float(loss.item()) * len(target)
                validation_count += len(target)

        train_loss = train_sum / max(train_count, 1)
        validation_loss = validation_sum / max(validation_count, 1)
        scheduler.step(validation_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"    epoch {epoch:3d}: train={train_loss:.6f}, val={validation_loss:.6f}")
        if stale_epochs >= config.patience:
            print(f"    early stop at epoch {epoch}; best validation loss={best_loss:.6f}")
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict(
    model: nn.Module,
    loader: DataLoader,
    samples: Sequence[Sample],
    stations: Sequence[str],
    target_stats: Mapping[int, Tuple[float, float]],
    device: torch.device,
) -> pd.DataFrame:
    outputs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for sequence, station_index, _ in loader:
            outputs.append(model(sequence.to(device), station_index.to(device)).cpu().numpy())
    predictions = np.concatenate(outputs) if outputs else np.array([], dtype=float)
    records = []
    for sample, value in zip(samples, predictions):
        if sample.station_index in target_stats:
            mean, std = target_stats[sample.station_index]
            value = value * std + mean
        records.append({
            "Timestamp": sample.timestamp,
            "substation": stations[sample.station_index],
            "actual_kw": sample.target,
            "predicted_kw": max(0.0, float(value)),
        })
    return pd.DataFrame(records)


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    error = actual - predicted
    abs_error = np.abs(error)
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    nonzero = np.abs(actual) > 1e-6
    mape = float(np.mean(abs_error[nonzero] / np.abs(actual[nonzero])) * 100) if nonzero.any() else math.nan
    smape_denominator = np.abs(actual) + np.abs(predicted)
    valid_smape = smape_denominator > 1e-6
    smape = float(np.mean(2 * abs_error[valid_smape] / smape_denominator[valid_smape]) * 100) if valid_smape.any() else math.nan
    wape_denominator = float(np.sum(np.abs(actual)))
    wape = float(np.sum(abs_error) / wape_denominator * 100) if wape_denominator > 1e-6 else math.nan
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - np.sum(error ** 2) / total) if total > 1e-12 else math.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "sMAPE_%": smape, "WAPE_%": wape, "R2": r2}


def calculate_metrics(predictions: pd.DataFrame, split: str, run: int, seed: int) -> pd.DataFrame:
    rows = []
    for station, group in predictions.groupby("substation", sort=True):
        group = group.dropna(subset=["actual_kw", "predicted_kw"])
        if group.empty:
            continue
        values = metric_values(group["actual_kw"].to_numpy(), group["predicted_kw"].to_numpy())
        rows.append({"run": run, "seed": seed, "split": split, "substation": station, **values})
    return pd.DataFrame(rows)


def save_wide_predictions(long_frame: pd.DataFrame, path: Path) -> None:
    wide = long_frame.pivot(index="Timestamp", columns="substation", values="predicted_kw")
    wide = wide.reindex(columns=TARGET_COLUMNS).reset_index()
    wide["Timestamp"] = pd.to_datetime(wide["Timestamp"]).dt.strftime("%m/%d/%Y %H:%M")
    wide.to_csv(path, index=False)


def svg_root(width: int, height: int) -> ET.Element:
    return ET.Element(
        "svg",
        {"xmlns": "http://www.w3.org/2000/svg", "width": str(width), "height": str(height), "viewBox": f"0 0 {width} {height}"},
    )


def svg_text(root: ET.Element, x: float, y: float, value: str, size: int = 12, anchor: str = "start") -> None:
    node = ET.SubElement(root, "text", {"x": f"{x:.1f}", "y": f"{y:.1f}", "font-size": str(size), "text-anchor": anchor, "font-family": "Arial, sans-serif", "fill": "#172033"})
    node.text = value


def write_svg(root: ET.Element, path: Path) -> None:
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def plot_training_curve(history: pd.DataFrame, path: Path, title: str) -> None:
    width, height = 900, 480
    root = svg_root(width, height)
    ET.SubElement(root, "rect", {"width": str(width), "height": str(height), "fill": "white"})
    svg_text(root, width / 2, 30, title, 18, "middle")
    left, top, right, bottom = 70, 55, 870, 420
    ET.SubElement(root, "line", {"x1": str(left), "y1": str(bottom), "x2": str(right), "y2": str(bottom), "stroke": "#64748b"})
    ET.SubElement(root, "line", {"x1": str(left), "y1": str(top), "x2": str(left), "y2": str(bottom), "stroke": "#64748b"})
    all_values = np.concatenate([history["train_loss"].to_numpy(), history["validation_loss"].to_numpy()])
    positive = all_values[all_values > 0]
    low = float(positive.min()) if len(positive) else 1e-6
    high = float(positive.max()) if len(positive) else 1.0
    low_log, high_log = math.log10(low), math.log10(high)
    if abs(high_log - low_log) < 1e-9:
        high_log += 1
    def points(values: Iterable[float]) -> str:
        result = []
        count = max(len(history) - 1, 1)
        for index, value in enumerate(values):
            x = left + (right - left) * index / count
            y = bottom - (bottom - top) * (math.log10(max(float(value), low)) - low_log) / (high_log - low_log)
            result.append(f"{x:.1f},{y:.1f}")
        return " ".join(result)
    ET.SubElement(root, "polyline", {"points": points(history["train_loss"]), "fill": "none", "stroke": "#2563eb", "stroke-width": "2"})
    ET.SubElement(root, "polyline", {"points": points(history["validation_loss"]), "fill": "none", "stroke": "#dc2626", "stroke-width": "2"})
    svg_text(root, left, 452, "Epoch", 12)
    svg_text(root, 10, top, "Loss (log scale)", 12)
    svg_text(root, right - 160, top + 15, "Train", 12)
    ET.SubElement(root, "line", {"x1": str(right - 200), "y1": str(top + 10), "x2": str(right - 170), "y2": str(top + 10), "stroke": "#2563eb", "stroke-width": "3"})
    svg_text(root, right - 70, top + 15, "Validation", 12)
    ET.SubElement(root, "line", {"x1": str(right - 115), "y1": str(top + 10), "x2": str(right - 80), "y2": str(top + 10), "stroke": "#dc2626", "stroke-width": "3"})
    write_svg(root, path)


def plot_forecast(predictions: pd.DataFrame, path: Path, title: str) -> None:
    stations = TARGET_COLUMNS
    width, panel_height = 1200, 185
    height = 65 + panel_height * len(stations)
    root = svg_root(width, height)
    ET.SubElement(root, "rect", {"width": str(width), "height": str(height), "fill": "white"})
    svg_text(root, width / 2, 30, title, 20, "middle")
    left, right = 80, width - 25
    for panel, station in enumerate(stations):
        group = predictions[predictions["substation"] == station].sort_values("Timestamp")
        if group.empty:
            continue
        top = 55 + panel * panel_height
        bottom = top + panel_height - 35
        actual = group["actual_kw"].to_numpy()
        predicted = group["predicted_kw"].to_numpy()
        finite_actual = actual[np.isfinite(actual)]
        finite_predicted = predicted[np.isfinite(predicted)]
        plotted_values = np.concatenate([finite_actual, finite_predicted])
        minimum = float(plotted_values.min())
        maximum = float(plotted_values.max())
        span = max(maximum - minimum, 1.0)
        ET.SubElement(root, "rect", {"x": str(left), "y": str(top), "width": str(right-left), "height": str(bottom-top), "fill": "#f8fafc", "stroke": "#cbd5e1"})
        svg_text(root, left + 5, top + 18, station, 13)
        def line(values: np.ndarray) -> str:
            count = max(len(values) - 1, 1)
            return " ".join(
                f"{left + (right-left)*i/count:.1f},{bottom - (bottom-top)*(float(v)-minimum)/span:.1f}"
                for i, v in enumerate(values) if np.isfinite(v)
            )
        ET.SubElement(root, "polyline", {"points": line(actual), "fill": "none", "stroke": "#111827", "stroke-width": "1.2", "opacity": "0.8"})
        ET.SubElement(root, "polyline", {"points": line(predicted), "fill": "none", "stroke": "#e11d48", "stroke-width": "1.2", "opacity": "0.85"})
        svg_text(root, right - 160, top + 18, "Actual", 11)
        ET.SubElement(root, "line", {"x1": str(right-200), "y1": str(top+14), "x2": str(right-165), "y2": str(top+14), "stroke": "#111827", "stroke-width": "2"})
        svg_text(root, right - 70, top + 18, "Predicted", 11)
        ET.SubElement(root, "line", {"x1": str(right-110), "y1": str(top+14), "x2": str(right-75), "y2": str(top+14), "stroke": "#e11d48", "stroke-width": "2"})
    write_svg(root, path)


def plot_metrics(metrics: pd.DataFrame, path: Path, title: str) -> None:
    metrics_to_plot = ["MAE", "RMSE", "R2"]
    width, height = 1250, 760
    root = svg_root(width, height)
    ET.SubElement(root, "rect", {"width": str(width), "height": str(height), "fill": "white"})
    svg_text(root, width / 2, 30, title, 20, "middle")
    panel_width = 390
    colors = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2", "#4f46e5", "#be123c", "#65a30d"]
    for panel, metric in enumerate(metrics_to_plot):
        left = 55 + panel * panel_width
        top, bottom = 65, 660
        values = [float(metrics.loc[metrics["substation"] == station, metric].iloc[0]) for station in TARGET_COLUMNS]
        minimum = min(0.0, min(values)) if metric == "R2" else 0.0
        maximum = max(values)
        span = max(maximum - minimum, 1e-6)
        svg_text(root, left + 170, 55, metric, 17, "middle")
        ET.SubElement(root, "line", {"x1": str(left), "y1": str(bottom), "x2": str(left+340), "y2": str(bottom), "stroke": "#64748b"})
        bar_width = 27
        for index, (station, value) in enumerate(zip(TARGET_COLUMNS, values)):
            x = left + 8 + index * 37
            zero_y = bottom - (bottom-top) * (0-minimum) / span
            value_y = bottom - (bottom-top) * (value-minimum) / span
            y = min(zero_y, value_y)
            h = max(abs(zero_y-value_y), 1)
            ET.SubElement(root, "rect", {"x": str(x), "y": f"{y:.1f}", "width": str(bar_width), "height": f"{h:.1f}", "fill": colors[index]})
            label = station.replace("(kW)", "")
            svg_text(root, x+bar_width/2, bottom+18, label, 9, "middle")
            svg_text(root, x+bar_width/2, max(top+12, y-4), f"{value:.2f}", 9, "middle")
    write_svg(root, path)


def model_factory(feature_count: int, station_count: int) -> nn.Module:
    return SimpleCNN1D(feature_count, station_count)


def train_one_group(
    config: ExperimentConfig,
    power: pd.DataFrame,
    scaled_features: np.ndarray,
    validation_days: set,
    group_name: str,
    run_index: int,
    seed: int,
    run_dir: Path,
    device: torch.device,
    factory: Callable[[int, int], nn.Module],
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stations = GROUPS[group_name]
    samples = build_samples(
        power, stations, config.lookback, validation_days,
        prediction_start, prediction_end,
    )
    normalize, criterion_factory = group_training_policy(group_name)
    stats = target_statistics(samples["train"], normalize)
    datasets = {
        split: SequenceDataset(scaled_features, split_samples, config.lookback, stats)
        for split, split_samples in samples.items()
    }
    loaders = {
        split: make_loader(dataset, config, split == "train", seed)
        for split, dataset in datasets.items()
    }
    print(f"  {group_name}: train={len(datasets['train'])}, val={len(datasets['validation'])}, test={len(datasets['test'])}")
    model = factory(len(FEATURE_COLUMNS), len(stations)).to(device)
    model, history = train_model(model, loaders["train"], loaders["validation"], config, criterion_factory, device)

    group_dir = run_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "stations": stations,
        "feature_columns": FEATURE_COLUMNS,
        "lookback": config.lookback,
        "target_stats": stats,
        "seed": seed,
    }, group_dir / "best_model.pt")
    history.to_csv(group_dir / "training_history.csv", index=False)
    plot_training_curve(history, group_dir / "training_curve.svg", f"{ARCHITECTURE_LABEL}: run {run_index}, {group_name}")

    prediction_frames = []
    metric_frames = []
    for split in ["validation", "test"]:
        frame = predict(model, loaders[split], samples[split], stations, stats, device)
        frame.insert(0, "split", split)
        frame.to_csv(group_dir / f"{split}_predictions_long.csv", index=False)
        prediction_frames.append(frame)
        metric_frames.append(calculate_metrics(frame, split, run_index, seed))
    return pd.concat(prediction_frames, ignore_index=True), pd.concat(metric_frames, ignore_index=True)


ARCHITECTURE_LABEL = "Simple 1D CNN"
DEFAULT_OUTPUT_DIR = "outputs_cnn_1d_february_visible"


def run_experiment(
    config: ExperimentConfig,
    factory: Callable[[int, int], nn.Module] = model_factory,
    architecture_label: str = ARCHITECTURE_LABEL,
) -> None:
    global ARCHITECTURE_LABEL
    ARCHITECTURE_LABEL = architecture_label
    power_path = resolve_input_path(config.power_path)
    weather_path = resolve_input_path(config.weather_path)
    prediction_start, prediction_end = config.prediction_bounds()
    month_label = prediction_start.strftime("%B %Y")
    month_tag = config.month_tag()
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    power = load_power(power_path, prediction_start, prediction_end, config.lookback)
    covariates, seasonal_values = build_covariates(
        power, weather_path, prediction_start, prediction_end
    )
    pool_days = eligible_pool_days(
        power, config.lookback, prediction_start, prediction_end
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Architecture: {architecture_label}")
    print(f"Device: {device}")
    print(f"Power: {power_path}")
    print(f"Weather: {weather_path}")
    print(f"Prediction month: {month_label}")
    print("Prediction-month visibility: observed weather and targets included in training")
    print(f"All-data seasonal fallback values: {seasonal_values}")

    metadata = {
        **asdict(config),
        "seeds": list(config.seeds),
        "architecture": architecture_label,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "groups": dict(GROUPS),
        "reporting_start": str(prediction_start),
        "reporting_end_exclusive": str(prediction_end),
        "prediction_month_hidden": False,
        "evaluation_mode": "in_sample_reconstruction",
        "seasonal_climatology": seasonal_values,
        "power_path_resolved": str(power_path),
        "weather_path_resolved": str(weather_path),
        "device": str(device),
    }
    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    all_metrics = []
    all_test_predictions = []
    for run_index, seed in enumerate(config.seeds):
        print(f"\nRun {run_index + 1}/{len(config.seeds)} (seed={seed})")
        set_seed(seed)
        validation_days = choose_validation_days(pool_days, config.val_fraction, seed)
        in_validation = np.array([ts.normalize() in validation_days for ts in covariates.index])
        scaler_mask = ~in_validation
        scaler = StandardScaler().fit(covariates.loc[scaler_mask, FEATURE_COLUMNS])
        scaled_features = scaler.transform(covariates[FEATURE_COLUMNS]).astype(np.float32)

        run_dir = output_dir / f"run_{run_index:02d}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"validation_day": sorted(validation_days)}).to_csv(run_dir / "validation_days.csv", index=False)
        np.savez(
            run_dir / "feature_scaler.npz",
            mean=scaler.mean_, scale=scaler.scale_, feature_columns=np.array(FEATURE_COLUMNS),
        )
        run_predictions = []
        run_metrics = []
        for group_name in GROUPS:
            predictions, metrics = train_one_group(
                config, power, scaled_features, validation_days, group_name,
                run_index, seed, run_dir, device, factory,
                prediction_start, prediction_end,
            )
            run_predictions.append(predictions)
            run_metrics.append(metrics)
        combined_predictions = pd.concat(run_predictions, ignore_index=True)
        combined_metrics = pd.concat(run_metrics, ignore_index=True)
        combined_metrics.to_csv(run_dir / "metrics.csv", index=False)
        test_predictions = combined_predictions[combined_predictions["split"] == "test"].copy()
        save_wide_predictions(test_predictions, run_dir / f"{month_tag}_predictions.csv")
        # Stable FFNN-compatible filename in the ordinary wide power format.
        save_wide_predictions(test_predictions, run_dir / "predictions.csv")
        plot_forecast(
            test_predictions, run_dir / f"{month_tag}_forecast.svg",
            f"{architecture_label}: {month_label}, seed {seed}",
        )
        all_metrics.append(combined_metrics)
        test_predictions["run"] = run_index
        test_predictions["seed"] = seed
        all_test_predictions.append(test_predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(output_dir / "all_run_metrics.csv", index=False)
    summary = (
        metrics.groupby(["split", "substation"])[["MAE", "RMSE", "MAPE_%", "sMAPE_%", "WAPE_%", "R2"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "metrics_mean_std.csv", index=False)

    test_runs = pd.concat(all_test_predictions, ignore_index=True)
    ensemble = (
        test_runs.groupby(["Timestamp", "substation"], as_index=False)
        .agg(actual_kw=("actual_kw", "first"), predicted_kw=("predicted_kw", "mean"), prediction_std_kw=("predicted_kw", "std"))
    )
    ensemble.to_csv(output_dir / f"ensemble_{month_tag}_predictions_long.csv", index=False)
    save_wide_predictions(ensemble, output_dir / f"ensemble_{month_tag}_predictions.csv")
    # Top-level predictions.csv is the mean prediction across all seeded runs.
    save_wide_predictions(ensemble, output_dir / "predictions.csv")
    ensemble_metrics = calculate_metrics(ensemble, "test_ensemble", -1, -1)
    ensemble_metrics.to_csv(output_dir / f"ensemble_{month_tag}_metrics.csv", index=False)
    plot_forecast(
        ensemble, output_dir / f"ensemble_{month_tag}_forecast.svg",
        f"{architecture_label}: {len(config.seeds)}-seed ensemble, {month_label}",
    )
    if not ensemble_metrics.empty:
        plot_metrics(
            ensemble_metrics, output_dir / f"ensemble_{month_tag}_metrics.svg",
            f"{architecture_label}: {month_label} ensemble metrics",
        )
    print(f"\nFinished. Results saved to {output_dir}")


def parse_seeds(text: str) -> Tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def parse_args(default_output: str = DEFAULT_OUTPUT_DIR) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-path", default="data_power_final.csv")
    parser.add_argument("--weather-path", default="weather_final.csv")
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--prediction-year", type=int, default=2024)
    parser.add_argument("--prediction-month", type=int, default=10, choices=range(1, 13), metavar="1-12")
    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--val-fraction", type=float, default=0.40)
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS, help="Comma-separated seeds (default: 42,...,51)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.lookback < 2:
        parser.error("--lookback must be at least 2")
    if args.prediction_year < 1900:
        parser.error("--prediction-year must be 1900 or later")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive")
    return ExperimentConfig(**vars(args))


if __name__ == "__main__":
    run_experiment(parse_args())
