# Anchor

An initial non-neural cryptocurrency forecasting experiment: **RegimeAutoAnchor-v1**.
This adapts the candidate-anchor and shrinkage ideas from the earlier
`../ChronoLM/docs/auto_anchor.md` and
`../ChronoLM/src/chronolm/experiments/anchor_baseline.py`.
The implementation is standalone and requires only NumPy, pandas, and SciPy.

## Initial experiment

| Setting | Value |
|---|---|
| Target | Daily Binance spot closing price, USDT |
| Inputs | Univariate: each coin's own closing-price history |
| Sequence length | 90 daily observations |
| Prediction length | 7 daily prices, predicted together |
| Evaluation | Weekly walk-forward, stride 7, 52 complete windows per coin |
| Test targets | 2025-01-01 through 2025-12-30, UTC |
| Downloaded history | 2022-01-01 through 2025-12-31 |
| Calibration | Up to 104 prior, fully observed, nonoverlapping 7-day forecast windows |
| Metrics | Scaled MAE/RMSE in percent, raw MAE/RMSE in USDT, SMAPE in percent |
| Comparisons | Only the proposed model, as requested |

The **90-day sequence is the context for each candidate forecast**, not the
entire training history. Adaptive anchor weights use a separate historical
calibration archive, with its own 90-day contexts and completed 7-day targets.
At the first evaluation origin, all calibration outcomes predate 2025.
Later origins may use earlier evaluation outcomes once fully observed; this is
an online walk-forward protocol. Hyperparameters are fixed before the run.
The last day of 2025 is unscored to keep every forecast block complete and disjoint.

The 30 fixed pairs are BTC, ETH, BNB, XRP, ADA, DOGE, SOL, DOT, LTC, BCH, LINK,
XLM, ATOM, ETC, TRX, XTZ, VET, FIL, THETA, UNI, AAVE, NEAR, ALGO, AVAX, EGLD,
SAND, MANA, AXS, CRV, and RUNE, all quoted in USDT. This is a convenience
universe of familiar coins with long histories, not a historical top-30 ranking.
It does not eliminate survivorship bias.

## Run

From `/Users/admin/LG/Anchor`, create an environment if needed:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m anchor.data
.venv/bin/python -m anchor.evaluate --output results/initial_2025_scaled
```

The first local run uses the already installed interpreter
`/Users/admin/LG/ChronoLM/.venv/bin/python`, with exactly the dependency versions
in `requirements.txt`; no changes to the ChronoLM project are necessary.
You can substitute that interpreter in the commands above.

The downloader uses Binance's public market-data endpoint without an API key.
It validates continuous daily coverage, timestamps, closed candles, and prices;
it fails on missing data instead of silently filling gaps or dropping coins.
Cached data must match the configured date range. Use a new `--data-dir` when
changing that range. Completed output directories cannot be overwritten;
choose another `--output` for a new run.

## Model

The original AutoAnchor selected a finite history anchor and optionally fitted
scalar shrinkage. Here, a convex statistical optimization learns a mixture of
14 candidate paths, conditioned on recent volatility and trend. A penalty pulls
weights toward the last-price anchor when the archive has weak effective support.
There are no neural networks, gradient-trained representations, external
covariates, or coin-specific rules. This is a proposed research extension;
novelty and superiority are not established by this initial run.

1. Build candidate paths from the last 90 closes: last price, recent means
   (3/7/14 days), EMAs (7/21), damped robust drifts (14/30/89 returns), local
   trends (14/30), gradual EMA reversion (two speeds), and ridge AR(7) returns.
2. Characterize each historical context using short/long volatility ratios
   and volatility-normalized 14/60-day cumulative returns.
3. Weight matured calibration windows by exponential recency (26-window
   half-life) and similarity to the current state.
4. Fit nonnegative mixture weights summing to one, minimizing volatility- and
   horizon-scaled squared price errors plus a persistence regularizer.
5. Apply those weights to the current candidate paths to predict all 7 closes.

For calibration window `i`, horizon `h`, and candidate `k`, let
`X[i,h,k] = (candidate[i,h,k] - last[i]) / (last[i] * sigma[i] * sqrt(h))`,
and define `y[i,h]` similarly from the observed target. Minimize

```text
sum_i p_i * mean_h (sum_k X[i,h,k] * w_k - y[i,h])^2
    + lambda * ||w - e_last||^2
subject to w >= 0 and sum(w) = 1.

p_i ∝ 2^(-age_i / 26) * (0.2 + 0.8 * exp(-mean((state_i-state_now)^2) / 2))
effective_n = 1 / sum_i p_i^2
lambda = 0.1 * sqrt(number_of_windows / effective_n)
```

`sigma` is the sample standard deviation of log returns in the observed context,
floored at 0.0001. State coordinates are clipped to [-3, 3]. Candidate log-price
displacements are bounded by `4 * sigma * sqrt(h)` using past information only.
The same weight vector applies to the full path; candidate paths themselves vary
by horizon. All constants are fixed in code/config, without selection on 2025 errors.
MAE and SMAPE are evaluation metrics; fitting optimizes normalized squared error.

## Outputs

Under `results/initial_2025_scaled/`:

- `REPORT.md`: readable results for all 30 coins.
- `per_coin_metrics.csv`: scaled MAE/RMSE, raw MAE/RMSE, SMAPE, and mean actual close over all 364 targets for each coin.
- `predictions.csv`: every actual/predicted price, forecast date, and horizon.
- `fold_metrics.csv`: weekly errors by coin.
- `per_coin_horizon_metrics.csv`: each coin's errors at steps 1 through 7.
- `horizon_metrics.csv`: pooled errors by forecast step, including scaled and raw MAE/RMSE.
- `calibration.jsonl`: per-origin fitted weights and calibration timing audit.
- `config.json`, `manifest.json`: configuration, input/source hashes, runtime versions.
- `summary.json`: aggregate figures and timing.

RMSE is computed from all individual squared errors within each coin, not by
averaging fold RMSEs. SMAPE is `mean(200*abs(actual-predicted)/(abs(actual)+abs(predicted)))`,
with zero contribution if both values are zero. `MAE_pct` is
`100*MAE/mean(abs(actual))`, and `RMSE_pct` is
`100*RMSE/mean(abs(actual))`, where the mean actual value is computed over the
same rows being scored. Equal-coin mean scaled MAE/RMSE and SMAPE are the main
cross-coin summaries; pooled raw-price errors can be dominated by expensive coins.

## TSLib top-10 benchmark

The project includes a benchmark wrapper for ten Time-Series-Library models:
TimeXer, TimeMixer, iTransformer, PatchTST, TimesNet, DLinear,
Nonstationary_Transformer, FEDformer, Autoformer, and Informer. The external
repository is cloned under `external/Time-Series-Library` and is imported
directly; no local model files are copied from it.

The benchmark keeps the same crypto task shape as the initial run:

- one univariate model per coin;
- 90 daily closes as input;
- 7 daily closes as output;
- the same 30 Binance USDT pairs;
- the same 52 weekly 2025 target blocks;
- MAE/RMSE in raw USDT and scaled percent, plus SMAPE percent.

By default, each neural model is trained once per coin using only pre-2025
windows, with a pre-2025 validation tail for early stopping, then evaluated on
the fixed 2025 weekly origins. Add `--refit-each-fold` only if you want the much
more expensive strict online version that retrains at every weekly origin.

Install the benchmark dependencies on the GPU server. If
`external/Time-Series-Library` is not present after moving the project, clone it
first:

```sh
cd /Users/admin/LG/Anchor
git clone https://github.com/thuml/Time-Series-Library.git external/Time-Series-Library
```

Then install:

```sh
python3 -m venv .venv-tslib
.venv-tslib/bin/python -m pip install -r requirements-tslib-benchmark.txt
```

Run the full benchmark:

```sh
.venv-tslib/bin/python -m anchor.benchmark_tslib \
  --device cuda \
  --output results/tslib_top10_2025
```

Useful debug commands:

```sh
.venv-tslib/bin/python -m anchor.benchmark_tslib \
  --import-check-only \
  --output results/tslib_import_check

.venv-tslib/bin/python -m anchor.benchmark_tslib \
  --models DLinear \
  --symbols BTCUSDT \
  --device cpu \
  --max-train-batches 1 \
  --max-val-batches 1 \
  --max-test-batches 1 \
  --output results/smoke_tslib_dlinear
```

Outputs under the benchmark run directory:

- `REPORT.md`: model ranking by equal-coin mean SMAPE.
- `per_model_metrics.csv`: macro metrics by model.
- `per_coin_metrics.csv`: per-model, per-coin metrics.
- `predictions.csv`: every 2025 target date prediction.
- `runs.csv`: training windows, validation loss, epoch count, and parameters.
- `import_check.csv`: TSLib import/dependency check.
- `manifest.json`: benchmark settings, metric definitions, and TSLib commit.

## Sources and research context

- [Earlier AutoAnchor method](../ChronoLM/docs/auto_anchor.md).
- [Binance public market-data endpoint documentation](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md).
- [Binance kline REST specification](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data).
- [Puoti et al., Quantifying Cryptocurrency Unpredictability (2025)](https://arxiv.org/abs/2502.09079)
  studies the difficulty of cryptocurrency price forecasting and reports strong
  naive-model performance. It motivates including a persistence anchor within
  this model; it does not establish performance of this adaptation.

Only this model is evaluated now. A later benchmark and ablation study would be
needed to support comparative research claims. If results motivate changes,
keep this run and treat 2025 as development data for those subsequent changes;
evaluate the revised method on a fresh held-out period.
