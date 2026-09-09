"""Regime-conditioned, persistence-regularized convex AutoAnchor adaptation.

The 90-day candidate context and the separate historical calibration archive
are intentionally distinct. Every archived target must mature before prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


def ema(x, span):
    value = x[0]
    alpha = 2 / (span + 1)
    for item in x[1:]:
        value += alpha * (item - value)
    return value


def describe(history):
    r = np.diff(np.log(history))
    sigma = max(float(np.std(r, ddof=1)), 1e-4)
    state = np.array([
        np.log(max(np.std(r[-7:], ddof=1), 1e-4) / sigma),
        np.log(max(np.std(r[-30:], ddof=1), 1e-4) / sigma),
        np.sum(r[-14:]) / (sigma * np.sqrt(14)),
        np.sum(r[-60:]) / (sigma * np.sqrt(60)),
    ])
    return sigma, np.clip(state, -3, 3)


def candidates(history, horizon):
    history = np.asarray(history, dtype=float)
    if len(history) < 90 or horizon < 1 or not np.isfinite(history).all() or (history <= 0).any():
        raise ValueError("Need at least 90 positive finite observations and a positive horizon")
    z = np.log(history)
    r = np.diff(z)
    sigma, state = describe(history)
    h = np.arange(1, horizon + 1)
    forecasts = {"last": np.full(horizon, z[-1])}
    for window in (3, 7, 14):
        forecasts[f"mean{window}"] = np.full(horizon, np.log(history[-window:].mean()))
    for span in (7, 21):
        forecasts[f"ema{span}"] = np.full(horizon, ema(z, span))
    damping = np.cumsum(0.9 ** np.arange(horizon))
    for window in (14, 30, 89):
        recent = r[-window:]
        lo, hi = np.quantile(recent, [0.05, 0.95])
        drift = np.mean(np.clip(recent, lo, hi))
        forecasts[f"drift{window}"] = z[-1] + damping * drift
    for window in (14, 30):
        x = np.arange(window, dtype=float)
        slope = np.dot(x - x.mean(), z[-window:] - z[-window:].mean()) / np.sum((x - x.mean()) ** 2)
        forecasts[f"trend{window}"] = z[-1] + damping * slope
    for speed in (0.15, 0.35):
        forecasts[f"revert{speed}"] = z[-1] + (1 - (1 - speed) ** h) * (ema(z, 21) - z[-1])
    # Seven-lag ridge autoregression of locally standardized returns.
    rr = r / sigma
    lag = 7
    X = np.array([rr[t-lag:t][::-1] for t in range(lag, len(rr))])
    y = rr[lag:]
    beta = np.linalg.solve(X.T @ X + 10 * np.eye(lag), X.T @ y)
    past, steps = list(rr), []
    for _ in h:
        value = float(np.clip(beta @ np.asarray(past[-lag:][::-1]), -3, 3))
        past.append(value)
        steps.append(value * sigma)
    forecasts["ar7_ridge"] = z[-1] + np.cumsum(steps)
    # A history-only guard against unstable extrapolation; all candidates share it.
    logs = np.stack(list(forecasts.values()), axis=1)
    bound = 4 * sigma * np.sqrt(h[:, None])
    logs = z[-1] + np.clip(logs - z[-1], -bound, bound)
    return list(forecasts), np.exp(logs), sigma, state


@dataclass
class Record:
    origin: int
    last: float
    predictions: np.ndarray
    sigma: float
    state: np.ndarray
    target: np.ndarray


class RegimeAutoAnchor:
    def __init__(self, config):
        self.config = config
        self._candidate_cache = {}

    def _candidates(self, history, horizon):
        key = (history.tobytes(), horizon)
        if key not in self._candidate_cache:
            self._candidate_cache[key] = candidates(history, horizon)
        return self._candidate_cache[key]

    def predict(self, observed, return_audit=False):
        """Accept ONLY the observed price prefix, never the future price array."""
        observed = np.asarray(observed, dtype=float)
        L, H = self.config["sequence_length"], self.config["prediction_length"]
        if L < 90 or H < 1 or len(observed) < L + H:
            raise ValueError("Insufficient history or invalid context/horizon")
        names, current, sigma, state = self._candidates(observed[-L:], H)
        # Weekly disjoint target blocks ending at or before len(observed).
        origins = list(range(len(observed) - H, L - 1, -H))[:self.config["calibration_windows"]]
        records = []
        for origin in reversed(origins):
            _, pred, vol, features = self._candidates(observed[origin-L:origin], H)
            target = observed[origin:origin+H]
            assert origin + H <= len(observed)
            records.append(Record(origin, observed[origin-1], pred, vol, features, target))
        if len(records) < 12:
            raise ValueError("Need at least 12 fully observed calibration windows")
        age = np.arange(len(records)-1, -1, -1)
        recency = 0.5 ** (age / self.config["calibration_half_life_windows"])
        distance = np.mean((np.stack([r.state for r in records]) - state) ** 2, axis=1)
        kernel = np.exp(-0.5 * distance / self.config["regime_bandwidth"] ** 2)
        probability = recency * (0.2 + 0.8 * kernel)
        probability /= probability.sum()
        effective_n = 1 / np.sum(probability ** 2)
        scale = np.array([r.last * r.sigma for r in records])[:, None] * np.sqrt(np.arange(1, H+1))
        last = np.array([r.last for r in records])[:, None]
        X = (np.stack([r.predictions for r in records]) - last[:, :, None]) / scale[:, :, None]
        y = (np.stack([r.target for r in records]) - last) / scale
        q = probability[:, None] / H
        A = np.einsum("nh,nhk,nhj->kj", np.broadcast_to(q, y.shape), X, X)
        b = np.einsum("nh,nhk,nh->k", np.broadcast_to(q, y.shape), X, y)
        prior = np.zeros(len(names))
        prior[0] = 1
        penalty = self.config["regularization"] * np.sqrt(len(records) / effective_n)

        def objective(w):
            return w @ A @ w - 2 * b @ w + penalty * np.sum((w-prior) ** 2)

        def gradient(w):
            return 2 * (A @ w - b + penalty * (w-prior))

        fit = minimize(objective, prior, jac=gradient, method="SLSQP",
                       bounds=[(0, 1)] * len(names),
                       constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                                    "jac": lambda w: np.ones_like(w)},
                       options={"ftol": 1e-10, "maxiter": 200})
        if not fit.success or not np.isfinite(fit.x).all():
            raise RuntimeError(f"Anchor weight optimization failed: {fit.message}")
        weights = np.maximum(fit.x, 0)
        weights /= weights.sum()
        prediction = current @ weights
        audit = dict(calibration_windows=len(records),
                     latest_calibration_target_index=records[-1].origin+H-1,
                     effective_windows=float(effective_n), regularization=float(penalty),
                     volatility=sigma, weights=dict(zip(names, map(float, weights))))
        return (prediction, audit) if return_audit else prediction
