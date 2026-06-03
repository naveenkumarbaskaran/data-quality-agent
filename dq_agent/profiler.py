"""DataProfiler: compute per-column statistics for DataFrames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DataProfiler:
    """Compute rich per-column statistics for a CSV or Parquet file."""

    # How many top-value counts to return per column
    TOP_K = 10

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def profile(self, path: str | Path) -> dict[str, Any]:
        """Load *path* and return a full profile dict.

        Parameters
        ----------
        path:
            Path to a ``.csv`` or ``.parquet`` file.

        Returns
        -------
        dict with keys:
            - ``file``   – absolute path string
            - ``shape``  – (rows, cols) tuple
            - ``columns`` – list of column names
            - ``dtypes``  – ``{col: dtype_string}``
            - ``stats``   – ``{col: per_column_stats}``
        """
        df = self._load(path)
        profile: dict[str, Any] = {
            "file": str(Path(path).resolve()),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "stats": {col: self._column_stats(df[col]) for col in df.columns},
        }
        return profile

    def detect_anomalies(
        self,
        path: str | Path,
        column: str,
    ) -> dict[str, Any]:
        """Detect anomalies in *column* of the dataset at *path*.

        Uses IQR-based outlier detection for numeric columns and
        frequency-based rarity detection for categorical columns.

        Returns
        -------
        dict with:
            - ``column``
            - ``dtype``
            - ``method`` – detection algorithm used
            - ``anomalies`` – list of (index, value) pairs
            - ``summary`` – human-readable summary
        """
        df = self._load(path)
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. Available: {list(df.columns)}"
            )
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            return self._numeric_anomalies(series)
        return self._categorical_anomalies(series)

    def compare_schemas(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        """Return schema-drift info between a baseline and current profile."""
        baseline_dtypes: dict[str, str] = baseline.get("dtypes", {})
        current_dtypes: dict[str, str] = current.get("dtypes", {})

        added = [c for c in current_dtypes if c not in baseline_dtypes]
        removed = [c for c in baseline_dtypes if c not in current_dtypes]
        type_changed = [
            {
                "column": col,
                "before": baseline_dtypes[col],
                "after": current_dtypes[col],
            }
            for col in baseline_dtypes
            if col in current_dtypes and baseline_dtypes[col] != current_dtypes[col]
        ]
        shape_changed = baseline.get("shape") != current.get("shape")

        drift_detected = bool(added or removed or type_changed or shape_changed)
        return {
            "drift_detected": drift_detected,
            "added_columns": added,
            "removed_columns": removed,
            "type_changes": type_changed,
            "shape_before": baseline.get("shape"),
            "shape_after": current.get("shape"),
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load(path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        suffix = p.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(p)
        if suffix in (".csv", ".tsv", ".txt"):
            sep = "\t" if suffix in (".tsv", ".txt") else ","
            return pd.read_csv(p, sep=sep)
        raise ValueError(f"Unsupported file type: {suffix}. Use .csv or .parquet")

    def _column_stats(self, series: pd.Series) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "dtype": str(series.dtype),
            "count": int(series.count()),
            "null_count": int(series.isna().sum()),
            "null_rate": round(float(series.isna().mean()), 6),
            "unique_count": int(series.nunique()),
        }

        # Top-value frequencies
        vc = series.value_counts(dropna=False).head(self.TOP_K)
        stats["top_values"] = {
            str(k): int(v) for k, v in vc.items()
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                stats.update(
                    {
                        "mean": _safe_float(clean.mean()),
                        "std": _safe_float(clean.std()),
                        "min": _safe_float(clean.min()),
                        "q25": _safe_float(clean.quantile(0.25)),
                        "median": _safe_float(clean.median()),
                        "q75": _safe_float(clean.quantile(0.75)),
                        "max": _safe_float(clean.max()),
                        "skewness": _safe_float(clean.skew()),
                        "kurtosis": _safe_float(clean.kurt()),
                    }
                )
            else:
                for key in ("mean", "std", "min", "q25", "median", "q75", "max",
                            "skewness", "kurtosis"):
                    stats[key] = None

        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # Character-length distribution for string columns
            lengths = series.dropna().astype(str).str.len()
            if len(lengths) > 0:
                stats["avg_length"] = _safe_float(lengths.mean())
                stats["max_length"] = int(lengths.max())
                stats["min_length"] = int(lengths.min())

        return stats

    @staticmethod
    def _numeric_anomalies(series: pd.Series) -> dict[str, Any]:
        clean = series.dropna()
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        anomalous = series[mask]
        anomalies = [
            {"index": int(idx), "value": _safe_float(val)}
            for idx, val in anomalous.items()
        ]
        return {
            "column": series.name,
            "dtype": str(series.dtype),
            "method": "IQR (1.5x)",
            "lower_bound": _safe_float(lower),
            "upper_bound": _safe_float(upper),
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:100],  # cap at 100 for readability
            "summary": (
                f"{len(anomalies)} outlier(s) found in '{series.name}' "
                f"(bounds: [{lower:.4g}, {upper:.4g}])"
            ),
        }

    @staticmethod
    def _categorical_anomalies(series: pd.Series) -> dict[str, Any]:
        """Flag values that appear in fewer than 1% of rows as rare."""
        total = len(series.dropna())
        if total == 0:
            return {
                "column": series.name,
                "dtype": str(series.dtype),
                "method": "frequency < 1%",
                "anomaly_count": 0,
                "anomalies": [],
                "summary": "No non-null values to analyse.",
            }
        threshold = 0.01 * total
        vc = series.value_counts()
        rare = vc[vc < threshold].index.tolist()
        mask = series.isin(rare)
        anomalous = series[mask]
        anomalies = [
            {"index": int(idx), "value": str(val)}
            for idx, val in anomalous.items()
        ]
        return {
            "column": series.name,
            "dtype": str(series.dtype),
            "method": "frequency < 1%",
            "rare_values": [str(v) for v in rare],
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:100],
            "summary": (
                f"{len(anomalies)} row(s) with rare values in '{series.name}' "
                f"({len(rare)} distinct rare value(s))"
            ),
        }


def _safe_float(value: Any) -> float | None:
    """Convert a numpy scalar to a Python float, returning None if not finite."""
    try:
        f = float(value)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
