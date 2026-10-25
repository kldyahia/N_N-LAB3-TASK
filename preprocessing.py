"""Train-only imputation and standardization for the selected two features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    """Median-impute missing values, then z-score using training data statistics."""

    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("Expected a 2-D feature matrix.")
        medians = np.nanmedian(x, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        imputed = self._impute_with(x, medians)
        means = np.mean(imputed, axis=0)
        scales = np.std(imputed, axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        self.medians = medians.astype(float)
        self.means = means.astype(float)
        self.scales = scales.astype(float)
        return self

    @staticmethod
    def _impute_with(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        missing_rows, missing_cols = np.where(~np.isfinite(x))
        if len(missing_rows):
            x[missing_rows, missing_cols] = medians[missing_cols]
        return x

    def impute(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._impute_with(x, self.medians)

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        imputed = self.impute(x)
        return (imputed - self.means) / self.scales

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(x, dtype=float) * self.scales + self.means

    def _check_fitted(self) -> None:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("Standardizer must be fitted before use.")
