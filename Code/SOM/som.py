
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SOM:


    def __init__(self, rows: int, cols: int, features: int, seed: int) -> None:
        self.rows, self.cols = rows, cols
        self.rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[:rows, :cols]
        self.locations = np.column_stack((yy.ravel(), xx.ravel()))
        self.weights = np.empty((rows, cols, features), dtype=float)

    def fit(
        self,
        data: np.ndarray,
        iterations: int,
        learning_rate: float,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        initial = self.rng.choice(len(data), self.rows * self.cols, replace=True)
        self.weights[:] = data[initial].reshape(self.weights.shape)
        checkpoints = np.unique(np.linspace(0, iterations - 1, min(100, iterations), dtype=int))
        history: list[float] = []
        history_steps: list[int] = []

        for step in range(iterations):
            sample = data[self.rng.integers(len(data))]
            distances = np.sum((self.weights - sample) ** 2, axis=2)
            winner = np.unravel_index(np.argmin(distances), distances.shape)
            progress = step / max(iterations - 1, 1)
            rate = learning_rate * math.exp(-2.5 * progress)
            radius = max(0.5, sigma * math.exp(-2.5 * progress))
            grid_distance = (
                (self.locations[:, 0] - winner[0]) ** 2
                + (self.locations[:, 1] - winner[1]) ** 2
            ).reshape(self.rows, self.cols)
            influence = np.exp(-grid_distance / (2.0 * radius**2))
            self.weights += rate * influence[..., None] * (sample - self.weights)

            if step in checkpoints:
                probe = data if len(data) <= 1500 else data[self.rng.choice(len(data), 1500, False)]
                _, errors, _ = self.map(probe)
                history_steps.append(step + 1)
                history.append(float(errors.mean()))
        return np.asarray(history_steps), np.asarray(history)

    def map(self, data: np.ndarray, chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat = self.weights.reshape(-1, self.weights.shape[-1])
        winners, errors, second_winners = [], [], []
        for start in range(0, len(data), chunk_size):
            batch = data[start : start + chunk_size]
            distances = np.sum((batch[:, None, :] - flat[None, :, :]) ** 2, axis=2)
            nearest_two = np.argpartition(distances, kth=1, axis=1)[:, :2]
            order = np.take_along_axis(distances, nearest_two, axis=1).argsort(axis=1)
            nearest_two = np.take_along_axis(nearest_two, order, axis=1)
            winners.append(nearest_two[:, 0])
            second_winners.append(nearest_two[:, 1])
            errors.append(np.sqrt(np.take_along_axis(distances, nearest_two[:, :1], axis=1)[:, 0]))
        return np.concatenate(winners), np.concatenate(errors), np.concatenate(second_winners)

    def u_matrix(self) -> np.ndarray:
        result = np.zeros((self.rows, self.cols))
        for row in range(self.rows):
            for col in range(self.cols):
                neighbors = self.weights[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2]
                distances = np.linalg.norm(neighbors - self.weights[row, col], axis=2).ravel()
                result[row, col] = distances[distances > 0].mean() if np.any(distances > 0) else 0
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=Path, default=Path("data_power_final.csv"))
    parser.add_argument("--weather", type=Path, default=Path("weather_final.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("som_outputs"))
    parser.add_argument("--rows", type=int, default=12, help="SOM grid rows")
    parser.add_argument("--cols", type=int, default=12, help="SOM grid columns")
    parser.add_argument("--iterations", type=int, default=15_000)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--anomaly-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(power_path: Path, weather_path: Path) -> tuple[pd.DataFrame, list[str], dict]:
    power = pd.read_csv(power_path)
    weather = pd.read_csv(weather_path)
    if "Timestamp" not in power or "date" not in weather:
        raise ValueError("Expected 'Timestamp' in power data and 'date' in weather data.")

    raw_power_rows = len(power)
    power["Timestamp"] = pd.to_datetime(power["Timestamp"], errors="coerce")
    invalid_timestamps = int(power["Timestamp"].isna().sum())
    power = power.dropna(subset=["Timestamp"]).copy()
    power_columns = [column for column in power.columns if column != "Timestamp"]
    power[power_columns] = power[power_columns].apply(pd.to_numeric, errors="coerce")
    duplicate_timestamps = int(power["Timestamp"].duplicated().sum())
    counts = power.groupby("Timestamp").size().rename("source_row_count")
    power = power.groupby("Timestamp", as_index=False)[power_columns].mean().join(counts, on="Timestamp")

    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.normalize()
    weather = weather.dropna(subset=["date"]).copy()
    weather_columns = [column for column in weather.columns if column != "date"]
    weather[weather_columns] = weather[weather_columns].apply(pd.to_numeric, errors="coerce")
    weather = weather.groupby("date", as_index=False)[weather_columns].mean()

    power["date"] = power["Timestamp"].dt.normalize()
    merged = power.merge(weather, on="date", how="inner", validate="many_to_one")
    dropped_without_weather = len(power) - len(merged)
    numeric_columns = power_columns + weather_columns
    all_missing = [column for column in numeric_columns if merged[column].isna().all()]
    if all_missing:
        raise ValueError(f"Columns contain no usable values: {all_missing}")
    missing_before = {column: int(merged[column].isna().sum()) for column in numeric_columns}
    merged[numeric_columns] = merged[numeric_columns].fillna(merged[numeric_columns].median())

    hour = merged["Timestamp"].dt.hour + merged["Timestamp"].dt.minute / 60.0
    weekday = merged["Timestamp"].dt.dayofweek
    merged["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    merged["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    merged["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    merged["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    feature_columns = numeric_columns + ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]

    audit = {
        "raw_power_rows": raw_power_rows,
        "invalid_power_timestamps_removed": invalid_timestamps,
        "duplicate_power_timestamps_averaged": duplicate_timestamps,
        "power_rows_without_weather_removed": dropped_without_weather,
        "training_rows": len(merged),
        "missing_values_median_imputed": missing_before,
    }
    return merged, feature_columns, audit


def save_heatmap(values: np.ndarray, path: Path, title: str, colorbar: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(values, cmap=cmap, origin="upper", aspect="equal")
    ax.set(title=title, xlabel="SOM column", ylabel="SOM row")
    fig.colorbar(image, ax=ax, shrink=0.8, label=colorbar)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_plots(
    output_dir: Path,
    som: SOM,
    features: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    winners: np.ndarray,
    errors: np.ndarray,
    anomalies: np.ndarray,
    history_steps: np.ndarray,
    history: np.ndarray,
) -> None:
    u_matrix = som.u_matrix()
    hits = np.bincount(winners, minlength=som.rows * som.cols).reshape(som.rows, som.cols)
    save_heatmap(u_matrix, output_dir / "u_matrix.png", "SOM U-matrix", "Mean neighbor distance")
    save_heatmap(np.log1p(hits), output_dir / "hit_map.png", "SOM hit map", "log(1 + samples)", "magma")

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(u_matrix, cmap="viridis", origin="upper")
    for row, col in np.argwhere(hits > 0):
        cluster_id = row * som.cols + col + 1
        ax.text(col, row, f"C{cluster_id}\n{hits[row, col]}", ha="center", va="center", fontsize=5,
                color="white" if u_matrix[row, col] > np.median(u_matrix) else "black")
    ax.set(title="Labeled SOM (cluster ID and sample count)", xlabel="SOM column", ylabel="SOM row")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Mean neighbor distance")
    fig.tight_layout()
    fig.savefig(output_dir / "labeled_som.png", dpi=200)
    plt.close(fig)

    original_weights = som.weights * scale + mean
    plot_cols = 4
    plot_rows = math.ceil(len(features) / plot_cols)
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(4 * plot_cols, 3.4 * plot_rows), squeeze=False)
    for index, (axis, feature) in enumerate(zip(axes.ravel(), features)):
        image = axis.imshow(original_weights[:, :, index], cmap="coolwarm", origin="upper")
        axis.set_title(feature, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.75)
    for axis in axes.ravel()[len(features) :]:
        axis.axis("off")
    fig.suptitle("SOM component maps", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "component_maps.png", dpi=180)
    plt.close(fig)

    rng = np.random.default_rng(123)
    rows, cols = np.divmod(winners, som.cols)
    fig, ax = plt.subplots(figsize=(10, 8))
    points = ax.scatter(cols + rng.normal(0, 0.12, len(cols)), rows + rng.normal(0, 0.12, len(rows)),
                        c=errors, cmap="plasma", s=7, alpha=0.45)
    ax.scatter(cols[anomalies], rows[anomalies], facecolors="none", edgecolors="cyan", s=45,
               linewidths=0.8, label="Anomaly")
    ax.invert_yaxis()
    ax.set(title="Mapped samples and anomalies", xlabel="SOM column", ylabel="SOM row")
    ax.legend(loc="upper right")
    fig.colorbar(points, ax=ax, label="Quantization error")
    fig.tight_layout()
    fig.savefig(output_dir / "sample_anomaly_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history_steps, history, linewidth=2)
    ax.set(title="SOM training curve", xlabel="Training iteration", ylabel="Mean quantization error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.rows < 2 or args.cols < 2 or args.iterations < 1:
        raise ValueError("Grid dimensions must be at least 2 and iterations must be positive.")
    if not 0 < args.anomaly_percentile < 100:
        raise ValueError("--anomaly-percentile must be between 0 and 100.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, features, audit = load_data(args.power, args.weather)
    matrix = data[features].to_numpy(dtype=float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (matrix - mean) / scale

    sigma = args.sigma if args.sigma is not None else max(args.rows, args.cols) / 2
    som = SOM(args.rows, args.cols, len(features), args.seed)
    history_steps, history = som.fit(normalized, args.iterations, args.learning_rate, sigma)
    winners, errors, second_winners = som.map(normalized)
    winner_rows, winner_cols = np.divmod(winners, args.cols)
    second_rows, second_cols = np.divmod(second_winners, args.cols)
    topographic_error = float(np.mean(np.maximum(abs(winner_rows - second_rows), abs(winner_cols - second_cols)) > 1))
    threshold = float(np.percentile(errors, args.anomaly_percentile))
    anomalies = errors >= threshold

    assignments = data.drop(columns=["date", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"])
    assignments = assignments.assign(
        som_row=winner_rows,
        som_col=winner_cols,
        cluster_id=winners + 1,
        quantization_error=errors,
        is_anomaly=anomalies,
    )
    assignments.to_csv(args.output_dir / "som_assignments.csv", index=False)

    summary = assignments.groupby(["cluster_id", "som_row", "som_col"], as_index=False).agg(
        sample_count=("Timestamp", "size"),
        mean_quantization_error=("quantization_error", "mean"),
        anomaly_count=("is_anomaly", "sum"),
        first_timestamp=("Timestamp", "min"),
        last_timestamp=("Timestamp", "max"),
    )
    feature_means = assignments.groupby("cluster_id")[features[:-4]].mean().add_prefix("mean_").reset_index()
    summary.merge(feature_means, on="cluster_id").to_csv(args.output_dir / "cluster_summary.csv", index=False)

    np.savez_compressed(
        args.output_dir / "som_model.npz",
        weights=som.weights,
        feature_mean=mean,
        feature_scale=scale,
        feature_names=np.asarray(features),
        grid_shape=np.asarray([args.rows, args.cols]),
        anomaly_threshold=threshold,
    )
    create_plots(args.output_dir, som, features, mean, scale, winners, errors, anomalies, history_steps, history)

    report = {
        **audit,
        "features": features,
        "grid": [args.rows, args.cols],
        "iterations": args.iterations,
        "seed": args.seed,
        "quantization_error_mean": float(errors.mean()),
        "quantization_error_median": float(np.median(errors)),
        "topographic_error": topographic_error,
        "anomaly_percentile": args.anomaly_percentile,
        "anomaly_threshold": threshold,
        "anomaly_count": int(anomalies.sum()),
        "occupied_clusters": int(assignments["cluster_id"].nunique()),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved SOM outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
