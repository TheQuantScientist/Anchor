import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from anchor.data import read_config, validate, DAY_MS
from anchor.evaluate import metrics
from anchor.model import RegimeAutoAnchor


CONFIG = read_config(Path(__file__).resolve().parents[1] / "configs/initial.json")


class ForecastTests(unittest.TestCase):
    def test_metrics_known_values(self):
        result = metrics([1, 2], [2, 2])
        self.assertAlmostEqual(result["MAE"], 0.5)
        self.assertAlmostEqual(result["RMSE"], np.sqrt(0.5))
        self.assertAlmostEqual(result["MAE_pct"], 100 * 0.5 / 1.5)
        self.assertAlmostEqual(result["RMSE_pct"], 100 * np.sqrt(0.5) / 1.5)
        self.assertAlmostEqual(result["mean_actual"], 1.5)
        self.assertAlmostEqual(result["SMAPE"], 100 / 3)
        self.assertEqual(metrics([0], [0])["SMAPE"], 0)
        self.assertEqual(metrics([0], [0])["MAE_pct"], 0)

    def test_constant_series_and_matured_labels(self):
        observed = np.full(900, 5.0)
        forecast, audit = RegimeAutoAnchor(CONFIG).predict(observed, True)
        np.testing.assert_allclose(forecast, 5)
        self.assertLess(audit["latest_calibration_target_index"], len(observed))
        self.assertEqual(audit["calibration_windows"], 104)
        self.assertAlmostEqual(sum(audit["weights"].values()), 1)

    def test_scale_equivariance(self):
        rng = np.random.default_rng(42)
        observed = np.exp(np.cumsum(rng.normal(0, 0.03, 900)))
        first = RegimeAutoAnchor(CONFIG).predict(observed)
        second = RegimeAutoAnchor(CONFIG).predict(observed * 10000)
        np.testing.assert_allclose(second, first * 10000, rtol=1e-7)

    def test_future_mutation_cannot_change_forecast(self):
        rng = np.random.default_rng(3)
        complete = np.exp(np.cumsum(rng.normal(0, 0.02, 920)))
        first = RegimeAutoAnchor(CONFIG).predict(complete[:900])
        complete[900:] *= 1000
        second = RegimeAutoAnchor(CONFIG).predict(complete[:900])
        np.testing.assert_array_equal(first, second)

    def test_missing_day_rejected(self):
        times = np.array([1577836800000, 1578009600000])
        frame = pd.DataFrame(dict(open_time=times, close_time=times+DAY_MS-1,
                                  open=[1, 1], high=[1, 1], low=[1, 1], close=[1, 1]))
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate(frame, "2020-01-01", "2020-01-04")


if __name__ == "__main__":
    unittest.main()
