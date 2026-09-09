"""Run one model on a locked, nonoverlapping walk-forward schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from .data import read_config, validate
from .model import RegimeAutoAnchor


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metric arrays must have identical nonempty shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Nonfinite metric input")
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mean_actual = float(np.mean(np.abs(actual)))
    if mean_actual > 0:
        mae_pct = 100 * mae / mean_actual
        rmse_pct = 100 * rmse / mean_actual
    elif mae == 0 and rmse == 0:
        mae_pct = rmse_pct = 0.0
    else:
        mae_pct = rmse_pct = float("inf")
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.divide(200 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0)
    return dict(MAE=mae, RMSE=rmse, MAE_pct=float(mae_pct), RMSE_pct=float(rmse_pct),
                SMAPE=float(np.mean(smape)), mean_actual=mean_actual)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config_path, data_dir, output):
    config = read_config(config_path)
    if config["stride"] != config["prediction_length"]:
        raise ValueError("Initial experiment requires stride equal to horizon")
    if config["interval"] != "1d" or len(set(config["symbols"])) != len(config["symbols"]):
        raise ValueError("Require daily interval and distinct symbols")
    output, data_dir = Path(output), Path(data_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    metadata = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), model=config["model"],
                    config_sha256=sha256(config_path), python=sys.version, platform=platform.platform(),
                    versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
                    source_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                    inputs=[], status="running")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start_time = time.perf_counter()
    all_predictions, all_folds, all_audits = [], [], []
    test_start = pd.Timestamp(config["test_start"], tz="UTC")
    test_end = pd.Timestamp(config["test_end_exclusive"], tz="UTC")
    H = config["prediction_length"]
    for symbol in config["symbols"]:
        path = data_dir / f"{symbol}_1d.csv"
        frame = pd.read_csv(path)
        validate(frame, config["data_start"], config["data_end_exclusive"])
        dates = pd.DatetimeIndex(pd.to_datetime(frame.open_time, unit="ms", utc=True))
        prices = frame.close.to_numpy(float)
        first, stop = dates.searchsorted(test_start), dates.searchsorted(test_end)
        if dates[first] != test_start or stop != first + (test_end-test_start).days:
            raise ValueError("Evaluation date range is not fully covered")
        model = RegimeAutoAnchor(config)
        coin_rows = []
        for fold, origin in enumerate(range(first, stop-H+1, config["stride"])):
            # Forecast is computed before accessing the target slice.
            forecast, audit = model.predict(prices[:origin], return_audit=True)
            actual = prices[origin:origin+H]
            observed_through = dates[origin-1].date().isoformat()
            target_start = dates[origin].date().isoformat()
            result = metrics(actual, forecast)
            all_folds.append(dict(symbol=symbol, fold=fold, observed_through=observed_through,
                                  target_start=target_start, **result))
            for step in range(H):
                row = dict(symbol=symbol, fold=fold, observed_through=observed_through,
                           target_date=dates[origin+step].date().isoformat(), horizon=step+1,
                           actual=float(actual[step]), predicted=float(forecast[step]))
                coin_rows.append(row)
            all_audits.append(dict(symbol=symbol, fold=fold, observed_through=observed_through,
                                   target_start=target_start, **audit))
        if not coin_rows:
            raise ValueError(f"No complete evaluation windows for {symbol}")
        all_predictions.extend(coin_rows)
        coin = pd.DataFrame(coin_rows)
        scores = metrics(coin.actual, coin.predicted)
        print(f"{symbol}: {len(coin)//H} folds | MAE%={scores['MAE_pct']:.4f} "
              f"RMSE%={scores['RMSE_pct']:.4f} SMAPE={scores['SMAPE']:.4f}%", flush=True)
        metadata["inputs"].append(dict(symbol=symbol, sha256=sha256(path), rows=len(frame)))
    predictions = pd.DataFrame(all_predictions)
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.12g")
    pd.DataFrame(all_folds).to_csv(output / "fold_metrics.csv", index=False)
    rows = []
    for symbol, group in predictions.groupby("symbol", sort=False):
        rows.append(dict(symbol=symbol, folds=group.fold.nunique(), predictions=len(group),
                         **metrics(group.actual, group.predicted)))
    per_coin = pd.DataFrame(rows)
    per_coin.to_csv(output / "per_coin_metrics.csv", index=False)
    horizons = [dict(horizon=int(h), **metrics(g.actual, g.predicted))
                for h, g in predictions.groupby("horizon")]
    pd.DataFrame(horizons).to_csv(output / "horizon_metrics.csv", index=False)
    coin_horizons = [dict(symbol=s, horizon=int(h), **metrics(g.actual, g.predicted))
                    for (s, h), g in predictions.groupby(["symbol", "horizon"], sort=False)]
    pd.DataFrame(coin_horizons).to_csv(output / "per_coin_horizon_metrics.csv", index=False)
    with (output / "calibration.jsonl").open("w") as stream:
        for audit in all_audits:
            stream.write(json.dumps(audit) + "\n")
    summary = dict(model=config["model"], coins=len(per_coin), folds_per_coin=int(per_coin.folds.iloc[0]),
                   total_predictions=len(predictions),
                   macro_MAE=float(per_coin.MAE.mean()), macro_RMSE=float(per_coin.RMSE.mean()),
                   macro_MAE_pct=float(per_coin.MAE_pct.mean()), macro_RMSE_pct=float(per_coin.RMSE_pct.mean()),
                   macro_SMAPE=float(per_coin.SMAPE.mean()), median_coin_SMAPE=float(per_coin.SMAPE.median()),
                   median_coin_MAE_pct=float(per_coin.MAE_pct.median()),
                   median_coin_RMSE_pct=float(per_coin.RMSE_pct.median()),
                   metric_units={"MAE": "USDT", "RMSE": "USDT",
                                 "MAE_pct": "100 * MAE / mean(abs(actual))",
                                 "RMSE_pct": "100 * RMSE / mean(abs(actual))",
                                 "SMAPE": "percent, range 0 to 200",
                                 "mean_actual": "USDT"},
                   scored_target_start=predictions.target_date.min(), scored_target_end=predictions.target_date.max(),
                   elapsed_seconds=time.perf_counter()-start_time)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Initial RegimeAutoAnchor results", "",
             f"90 daily closes → 7 daily closes; {summary['folds_per_coin']} weekly walk-forward folds per coin.",
             f"Scored targets: {summary['scored_target_start']} through {summary['scored_target_end']} (UTC).", "",
             "Scaled MAE/RMSE are normalized by each coin's mean actual close over the scored targets and reported as percentages.",
             "Raw MAE/RMSE remain in USDT for auditability. SMAPE is already a percentage (0–200).",
             "", f"**Equal-coin mean scaled MAE: {summary['macro_MAE_pct']:.4f}%.** "
             f"Mean scaled RMSE: {summary['macro_RMSE_pct']:.4f}%. "
             f"Mean SMAPE: {summary['macro_SMAPE']:.4f}%.", "",
             "| Coin | MAE (%) | RMSE (%) | SMAPE (%) | Mean actual | MAE (USDT) | RMSE (USDT) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['symbol']} | {row['MAE_pct']:.4f} | {row['RMSE_pct']:.4f} | "
                     f"{row['SMAPE']:.4f} | {row['mean_actual']:.6g} | "
                     f"{row['MAE']:.6g} | {row['RMSE']:.6g} |")
    lines += ["", "The final day of 2025 is excluded because it cannot form a complete 7-day block within the test year.",
              "No baselines were run. This initial experiment does not establish superiority or research novelty.",
              "The fixed coin universe is a convenience sample, not a historical market-cap ranking; survivorship bias remains.",
              "Past evaluation outcomes enter later calibration only after they have been observed, as in an online deployment.",
              "For cross-coin comparison, use MAE (%), RMSE (%), and SMAPE (%)."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    metadata.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=summary["elapsed_seconds"])
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/initial.json")
    parser.add_argument("--data-dir", default="data/binance_daily")
    parser.add_argument("--output", default="results/initial_2025")
    args = parser.parse_args()
    run(args.config, args.data_dir, args.output)
