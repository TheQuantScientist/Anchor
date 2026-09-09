"""Read-only Binance daily candle downloader with strict coverage checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
DAY_MS = 86_400_000
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_base_volume", "taker_quote_volume", "ignore"]


def read_config(path):
    return json.loads(Path(path).read_text())


def validate(frame, start, end):
    expected = pd.date_range(start, end, freq="D", inclusive="left", tz="UTC")
    dates = pd.to_datetime(frame.open_time, unit="ms", utc=True)
    if len(dates) != len(expected) or not np.array_equal(dates.to_numpy(), expected.to_numpy()):
        raise ValueError("Daily coverage is incomplete, duplicated, or out of order; no imputation allowed")
    prices = frame[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Invalid prices")
    if (frame.high < frame[["open", "close", "low"]].max(axis=1)).any() or (
        frame.low > frame[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Inconsistent OHLC prices")
    if not np.array_equal(frame.close_time.to_numpy(), frame.open_time.to_numpy() + DAY_MS - 1):
        raise ValueError("Unexpected candle closing timestamps")
    if (frame.close_time >= int(datetime.now(timezone.utc).timestamp() * 1000)).any():
        raise ValueError("Unclosed candle encountered")


def request(params):
    url = BASE_URL + "?" + urlencode(params)
    for attempt in range(5):
        try:
            with urlopen(url, timeout=30) as response:
                rows = json.load(response)
            if not isinstance(rows, list):
                raise ValueError(f"Unexpected Binance response: {rows}")
            return rows
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise
            time.sleep(min(30, max(2 ** attempt, int(exc.headers.get("Retry-After", "0")))))
        except (URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def download(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    start, end = config["data_start"], config["data_end_exclusive"]
    start_ms, end_ms = [int(pd.Timestamp(x, tz="UTC").timestamp() * 1000) for x in (start, end)]
    manifest = {"source": BASE_URL, "start": start, "end_exclusive": end,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    for symbol in config["symbols"]:
        path = directory / f"{symbol}_1d.csv"
        if path.exists():
            frame = pd.read_csv(path)
            validate(frame, start, end)
        else:
            rows, cursor = [], start_ms
            while cursor < end_ms:
                batch = request(dict(symbol=symbol, interval="1d", startTime=cursor,
                                     endTime=end_ms - 1, limit=1000))
                if not batch:
                    break
                next_cursor = int(batch[-1][0]) + DAY_MS
                if next_cursor <= cursor:
                    raise ValueError("Binance pagination did not advance")
                rows.extend(batch)
                cursor = next_cursor
                time.sleep(0.15)
            frame = pd.DataFrame(rows, columns=COLUMNS).apply(pd.to_numeric)
            validate(frame, start, end)
            temporary = path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False, float_format="%.15g")
            temporary.replace(path)
        manifest["files"].append(dict(symbol=symbol, rows=len(frame), path=path.name,
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(f"{symbol}: {len(frame)} daily candles verified", flush=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/initial.json")
    parser.add_argument("--data-dir", default="data/binance_daily")
    args = parser.parse_args()
    download(read_config(args.config), args.data_dir)
