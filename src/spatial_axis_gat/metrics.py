from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"n": int(mask.sum()), "pearson_r": np.nan, "spearman_r": np.nan, "mae": np.nan, "r2": np.nan}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "n": int(mask.sum()),
        "pearson_r": float(stats.pearsonr(yt, yp).statistic),
        "spearman_r": float(stats.spearmanr(yt, yp).statistic),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }


def summarize_metric_curve(df: pd.DataFrame, metric: str = "pearson_r") -> pd.DataFrame:
    return (
        df.groupby(["axis", "n_train_villi"], observed=True)[metric]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )

