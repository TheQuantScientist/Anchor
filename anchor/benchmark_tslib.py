"""Benchmark selected Time-Series-Library models on the locked crypto schedule."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .data import read_config, validate
from .evaluate import metrics


TOP10_MODELS = [
    "TimeXer",
    "TimeMixer",
    "iTransformer",
    "PatchTST",
    "TimesNet",
    "DLinear",
    "Nonstationary_Transformer",
    "FEDformer",
    "Autoformer",
    "Informer",
]


@dataclass(frozen=True)
class SeriesData:
    symbol: str
    dates: pd.DatetimeIndex
    prices: np.ndarray
    marks: np.ndarray


class WindowDataset:
    def __init__(self, values, marks, origins, seq_len, label_len, pred_len):
        self.values = values.astype(np.float32)
        self.marks = marks.astype(np.float32)
        self.origins = np.asarray(origins, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index])
        x0, x1 = origin - self.seq_len, origin
        y0, y1 = origin - self.label_len, origin + self.pred_len
        return (
            self.values[x0:x1],
            self.values[y0:y1],
            self.marks[x0:x1],
            self.marks[y0:y1],
            np.int64(origin),
        )


class Standardizer:
    def __init__(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.mean = float(values.mean())
        std = float(values.std(ddof=0))
        self.std = std if std > 1e-12 else 1.0

    def transform(self, values):
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse(self, values):
        return np.asarray(values, dtype=np.float64) * self.std + self.mean


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("Install PyTorch before running TSLib benchmarks") from exc
    return torch, DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch, _ = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_tslib_to_path(tslib_root):
    tslib_root = Path(tslib_root).resolve()
    if not (tslib_root / "models").is_dir():
        raise FileNotFoundError(f"Time-Series-Library models folder not found: {tslib_root}")
    root_text = str(tslib_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return tslib_root


def load_model_class(model_name, tslib_root):
    add_tslib_to_path(tslib_root)
    module = importlib.import_module(f"models.{model_name}")
    if not hasattr(module, "Model"):
        raise AttributeError(f"models.{model_name} does not expose a Model class")
    return module.Model


def time_marks(dates):
    dates = pd.DatetimeIndex(dates)
    return np.stack(
        [
            dates.dayofweek.to_numpy(dtype=np.float32) / 6.0 - 0.5,
            (dates.day.to_numpy(dtype=np.float32) - 1.0) / 30.0 - 0.5,
            (dates.dayofyear.to_numpy(dtype=np.float32) - 1.0) / 365.0 - 0.5,
        ],
        axis=1,
    )


def load_series(symbol, data_dir, config):
    path = Path(data_dir) / f"{symbol}_1d.csv"
    frame = pd.read_csv(path)
    validate(frame, config["data_start"], config["data_end_exclusive"])
    dates = pd.DatetimeIndex(pd.to_datetime(frame.open_time, unit="ms", utc=True))
    prices = frame.close.to_numpy(float)
    if not np.all(np.isfinite(prices)):
        raise ValueError(f"Nonfinite close price found for {symbol}")
    return SeriesData(symbol=symbol, dates=dates, prices=prices, marks=time_marks(dates))


def date_index(dates, iso_date, allow_exclusive_stop=False):
    stamp = pd.Timestamp(iso_date, tz="UTC")
    idx = int(dates.searchsorted(stamp))
    if allow_exclusive_stop and idx == len(dates):
        expected = dates[-1] + pd.Timedelta(days=1)
        if stamp == expected:
            return idx
    if idx >= len(dates) or dates[idx] != stamp:
        raise ValueError(f"Date {iso_date} is not covered")
    return idx


def make_origins(start, stop_exclusive, seq_len, pred_len, stride):
    lo = max(int(start), int(seq_len))
    hi = int(stop_exclusive) - int(pred_len) + 1
    if hi <= lo:
        return []
    return list(range(lo, hi, int(stride)))


def split_origins(series, bench_cfg, refit_origin=None):
    seq_len = bench_cfg["seq_len"]
    pred_len = bench_cfg["pred_len"]
    train_start = date_index(series.dates, bench_cfg["train_start"])
    test_start = date_index(series.dates, bench_cfg["test_start"])
    test_end = date_index(series.dates, bench_cfg["test_end_exclusive"], allow_exclusive_stop=True)

    if refit_origin is None:
        val_start = date_index(series.dates, bench_cfg["val_start"])
        train_origins = make_origins(train_start + seq_len, val_start, seq_len, pred_len, bench_cfg["train_stride"])
        val_origins = make_origins(val_start, test_start, seq_len, pred_len, bench_cfg["val_stride"])
    else:
        available = make_origins(train_start + seq_len, refit_origin, seq_len, pred_len, bench_cfg["train_stride"])
        val_count = max(1, int(round(0.15 * len(available)))) if len(available) >= 8 else 0
        train_origins = available[:-val_count] if val_count else available
        val_origins = available[-val_count:] if val_count else available[-1:]

    test_origins = make_origins(test_start, test_end, seq_len, pred_len, bench_cfg["test_stride"])
    if len(test_origins) != 52 and refit_origin is None:
        raise ValueError(f"Expected 52 complete weekly test folds, got {len(test_origins)}")
    if not train_origins or not val_origins:
        raise ValueError(f"Insufficient train/validation windows for {series.symbol}")
    return train_origins, val_origins, test_origins


def merged_model_config(base, overrides):
    cfg = dict(base)
    cfg.update(overrides or {})
    cfg.update(
        task_name="long_term_forecast",
        enc_in=1,
        dec_in=1,
        c_out=1,
        num_class=2,
        activation="gelu",
        distil=True,
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
    )
    cfg["features"] = "M"
    return cfg


def make_namespace(model_name, base, overrides, device):
    cfg = merged_model_config(base, overrides)
    cfg.update(model=model_name, use_gpu=device.type != "cpu", gpu=0, gpu_type=device.type, use_multi_gpu=False)
    return SimpleNamespace(**cfg)


def forward_model(model, batch, args, device):
    torch, _ = require_torch()
    batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    pred_len = int(args.pred_len)
    label_len = int(args.label_len)
    if label_len:
        dec_prefix = batch_y[:, :label_len, :]
        dec_zeros = torch.zeros_like(batch_y[:, -pred_len:, :])
        dec_inp = torch.cat([dec_prefix, dec_zeros], dim=1)
    else:
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :])
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    return outputs[:, -pred_len:, :], batch_y[:, -pred_len:, :]


def train_epoch(model, loader, optimizer, criterion, args, device, max_batches=None):
    model.train()
    losses = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        pred, true = forward_model(model, batch, args, device)
        loss = criterion(pred, true)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else math.inf


def evaluate_loss(model, loader, criterion, args, device, max_batches=None):
    torch, _ = require_torch()
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            pred, true = forward_model(model, batch, args, device)
            losses.append(float(criterion(pred, true).detach().cpu()))
    return float(np.mean(losses)) if losses else math.inf


def predict(model, loader, args, device, standardizer, series, max_batches=None):
    torch, _ = require_torch()
    model.eval()
    rows = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            pred, _ = forward_model(model, batch, args, device)
            pred = standardizer.inverse(pred.detach().cpu().numpy()[:, :, 0])
            origins = batch[-1].detach().cpu().numpy()
            for batch_i, origin in enumerate(origins):
                origin = int(origin)
                observed_through = series.dates[origin - 1].date().isoformat()
                for step in range(int(args.pred_len)):
                    idx = origin + step
                    rows.append(
                        dict(
                            symbol=series.symbol,
                            fold=None,
                            observed_through=observed_through,
                            target_date=series.dates[idx].date().isoformat(),
                            horizon=step + 1,
                            actual=float(series.prices[idx]),
                            predicted=float(pred[batch_i, step]),
                        )
                    )
    rows.sort(key=lambda r: (r["symbol"], r["target_date"], r["horizon"]))
    fold_by_start = {}
    for row in rows:
        key = row["target_date"] if row["horizon"] == 1 else None
        if key is not None and key not in fold_by_start:
            fold_by_start[key] = len(fold_by_start)
        if row["horizon"] == 1:
            row["fold"] = fold_by_start[row["target_date"]]
    current_fold = -1
    for row in rows:
        if row["horizon"] == 1:
            current_fold += 1
        row["fold"] = current_fold
    return rows


def train_one_model(model_name, series, standardizer, bench_cfg, overrides, args_cli, device, output_dir,
                    train_origins, val_origins, test_origins):
    torch, DataLoader = require_torch()
    values_scaled = standardizer.transform(series.prices)[:, None]
    cfg = make_namespace(model_name, bench_cfg, overrides, device)
    model_class = load_model_class(model_name, args_cli.tslib_root)
    model = model_class(cfg).float().to(device)
    train_dataset = WindowDataset(values_scaled, series.marks, train_origins, cfg.seq_len, cfg.label_len, cfg.pred_len)
    val_dataset = WindowDataset(values_scaled, series.marks, val_origins, cfg.seq_len, cfg.label_len, cfg.pred_len)
    test_dataset = WindowDataset(values_scaled, series.marks, test_origins, cfg.seq_len, cfg.label_len, cfg.pred_len)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = torch.nn.MSELoss()
    best_state = None
    best_val = math.inf
    stale = 0
    history = []
    for epoch in range(int(cfg.train_epochs)):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, cfg, device, args_cli.max_train_batches)
        val_loss = evaluate_loss(model, val_loader, criterion, cfg, device, args_cli.max_val_batches)
        history.append(dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    rows = predict(model, test_loader, cfg, device, standardizer, series, args_cli.max_test_batches)
    model_dir = output_dir / model_name / series.symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "train_history.json").write_text(json.dumps(history, indent=2) + "\n")
    torch.save(best_state or model.state_dict(), model_dir / "checkpoint.pt")
    return rows, dict(best_val_loss=best_val, epochs=len(history), train_windows=len(train_dataset),
                      val_windows=len(val_dataset), test_windows=len(test_dataset), parameters=count_parameters(model))


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def choose_device(requested):
    torch, _ = require_torch()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def parse_csv_arg(text, allowed):
    if text == "all":
        return list(allowed)
    values = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown values: {unknown}")
    return values


def write_outputs(output, all_predictions, run_rows, config, anchor_config, started_at, elapsed, tslib_root):
    predictions = pd.DataFrame(all_predictions)
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.12g")
    per_coin_rows = []
    for (model, symbol), group in predictions.groupby(["model", "symbol"], sort=False):
        per_coin_rows.append(dict(model=model, symbol=symbol, folds=group.fold.nunique(),
                                  predictions=len(group), **metrics(group.actual, group.predicted)))
    per_coin = pd.DataFrame(per_coin_rows)
    per_coin.to_csv(output / "per_coin_metrics.csv", index=False)
    per_model_rows = []
    for model, group in predictions.groupby("model", sort=False):
        coin_metric = per_coin[per_coin.model == model]
        per_model_rows.append(
            dict(
                model=model,
                coins=int(coin_metric.symbol.nunique()),
                folds_per_coin=int(coin_metric.folds.iloc[0]),
                total_predictions=int(len(group)),
                macro_MAE=float(coin_metric.MAE.mean()),
                macro_RMSE=float(coin_metric.RMSE.mean()),
                macro_MAE_pct=float(coin_metric.MAE_pct.mean()),
                macro_RMSE_pct=float(coin_metric.RMSE_pct.mean()),
                macro_SMAPE=float(coin_metric.SMAPE.mean()),
                median_coin_SMAPE=float(coin_metric.SMAPE.median()),
            )
        )
    per_model = pd.DataFrame(per_model_rows).sort_values(["macro_SMAPE", "macro_MAE_pct"])
    per_model.to_csv(output / "per_model_metrics.csv", index=False)
    pd.DataFrame(run_rows).to_csv(output / "runs.csv", index=False)
    metadata = dict(
        status="complete",
        started_at_utc=started_at,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=elapsed,
        tslib_root=str(Path(tslib_root).resolve()),
        tslib_git_commit=git_commit(tslib_root),
        config=config,
        anchor_config=anchor_config,
        metric_units={
            "MAE": "USDT",
            "RMSE": "USDT",
            "MAE_pct": "100 * MAE / mean(abs(actual))",
            "RMSE_pct": "100 * RMSE / mean(abs(actual))",
            "SMAPE": "percent, range 0 to 200",
            "mean_actual": "USDT",
        },
    )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    lines = [
        "# TSLib top-10 benchmark",
        "",
        "90 daily closes -> 7 daily closes; weekly 2025 walk-forward targets.",
        "Models are sorted by equal-coin mean SMAPE.",
        "",
        "| Rank | Model | MAE (%) | RMSE (%) | SMAPE (%) |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(per_model.itertuples(index=False), start=1):
        lines.append(f"| {rank} | {row.model} | {row.macro_MAE_pct:.4f} | "
                     f"{row.macro_RMSE_pct:.4f} | {row.macro_SMAPE:.4f} |")
    lines.extend(["", "Raw USDT metrics and per-coin rows are in the CSV files."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def git_commit(path):
    git = shutil.which("git")
    if not git:
        return None
    import subprocess

    result = subprocess.run([git, "-C", str(path), "rev-parse", "HEAD"], check=False,
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def smoke_imports(models, tslib_root):
    rows = []
    for model_name in models:
        try:
            load_model_class(model_name, tslib_root)
            rows.append(dict(model=model_name, ok=True, error=""))
        except Exception as exc:
            rows.append(dict(model=model_name, ok=False, error=f"{type(exc).__name__}: {exc}"))
    return rows


def run(args_cli):
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    bench_file = Path(args_cli.benchmark_config)
    config = json.loads(bench_file.read_text())
    anchor_config = read_config(args_cli.anchor_config)
    models = parse_csv_arg(args_cli.models, TOP10_MODELS)
    symbols = parse_csv_arg(args_cli.symbols, anchor_config["symbols"])
    output = Path(args_cli.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "anchor_config.json").write_text(json.dumps(anchor_config, indent=2) + "\n")
    add_tslib_to_path(args_cli.tslib_root)
    import_rows = smoke_imports(models, args_cli.tslib_root)
    pd.DataFrame(import_rows).to_csv(output / "import_check.csv", index=False)
    bad = [row for row in import_rows if not row["ok"]]
    if bad:
        message = "; ".join(f"{row['model']}: {row['error']}" for row in bad)
        if args_cli.import_check_only:
            print(message)
            return
        raise RuntimeError(f"TSLib import check failed. Install missing dependencies. {message}")
    if args_cli.import_check_only:
        print(f"Import check passed for {len(import_rows)} models")
        return
    torch, _ = require_torch()
    set_seed(int(config["base"]["seed"]))
    device = choose_device(args_cli.device)
    all_predictions, run_rows = [], []
    for symbol in symbols:
        series = load_series(symbol, args_cli.data_dir, anchor_config)
        base_train_origins, base_val_origins, base_test_origins = split_origins(series, config["base"])
        scale_stop = base_val_origins[0] if base_val_origins else date_index(series.dates, config["base"]["val_start"])
        standardizer = Standardizer(series.prices[:scale_stop])
        for model_name in models:
            overrides = config["models"].get(model_name, {})
            if args_cli.refit_each_fold:
                model_predictions = []
                stats = []
                for origin in base_test_origins:
                    train_origins, val_origins, _ = split_origins(series, config["base"], refit_origin=origin)
                    fold_scaler = Standardizer(series.prices[:origin])
                    rows, info = train_one_model(model_name, series, fold_scaler, config["base"], overrides,
                                                 args_cli, device, output, train_origins, val_origins, [origin])
                    model_predictions.extend(rows)
                    stats.append(info)
                info = dict(best_val_loss=float(np.mean([s["best_val_loss"] for s in stats])),
                            epochs=float(np.mean([s["epochs"] for s in stats])),
                            train_windows=float(np.mean([s["train_windows"] for s in stats])),
                            val_windows=float(np.mean([s["val_windows"] for s in stats])),
                            test_windows=len(model_predictions) // config["base"]["pred_len"],
                            parameters=stats[-1]["parameters"])
                rows = model_predictions
            else:
                rows, info = train_one_model(model_name, series, standardizer, config["base"], overrides, args_cli,
                                             device, output, base_train_origins, base_val_origins, base_test_origins)
            for row in rows:
                row["model"] = model_name
            scores = metrics([row["actual"] for row in rows], [row["predicted"] for row in rows])
            run_rows.append(dict(model=model_name, symbol=symbol, **info, **scores))
            all_predictions.extend(rows)
            print(f"{model_name} {symbol}: MAE%={scores['MAE_pct']:.4f} "
                  f"RMSE%={scores['RMSE_pct']:.4f} SMAPE={scores['SMAPE']:.4f}%")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not all_predictions:
        raise RuntimeError("No predictions were produced")
    write_outputs(output, all_predictions, run_rows, config, anchor_config, started_at,
                  time.perf_counter() - start, args_cli.tslib_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-config", default="configs/initial.json")
    parser.add_argument("--benchmark-config", default="configs/tslib_top10.json")
    parser.add_argument("--data-dir", default="data/binance_daily")
    parser.add_argument("--tslib-root", default="external/Time-Series-Library")
    parser.add_argument("--output", default="results/tslib_top10_2025")
    parser.add_argument("--models", default="all", help="Comma-separated model list or 'all'")
    parser.add_argument("--symbols", default="all", help="Comma-separated symbol list or 'all'")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--refit-each-fold", action="store_true",
                        help="Retrain at each weekly origin using only data observed by that origin")
    parser.add_argument("--import-check-only", action="store_true",
                        help="Only verify TSLib model imports and write import_check.csv")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Debug/smoke-test limit; leave unset for real runs")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Debug/smoke-test limit; leave unset for real runs")
    parser.add_argument("--max-test-batches", type=int, default=None,
                        help="Debug/smoke-test limit; leave unset for real runs")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
