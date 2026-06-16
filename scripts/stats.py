"""
Statistical-testing helpers for the ACID_Journal paper.

Two main entry points:

  bootstrap_ci(metric_fn, *args, n=1000, alpha=0.05)
      Nonparametric percentile bootstrap 95% CI over the test set.

  mcnemar(labels, preds_a, preds_b)
      Paired McNemar test for two classifiers' predictions on the same test
      set. Returns (b, c, p_value).

  paired_permutation(labels, preds_a, preds_b, metric_fn, n=10000)
      Generic paired permutation test on any metric.

These are used by `aggregate.py` to build the headline tables and p-value
matrices for the paper.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (point_estimate, lo, hi) for `metric_fn(y_true, y_pred)`."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    point = float(metric_fn(y_true, y_pred))
    boots = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        try:
            boots[i] = float(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            boots[i] = np.nan
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def mcnemar(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> Tuple[int, int, float]:
    """Paired McNemar's test with continuity correction.

    b = samples where A is correct and B is wrong.
    c = samples where A is wrong and B is correct.
    p = chi-square p-value on 1 d.f. (or binomial exact if b+c < 25).
    """
    from scipy.stats import binom, chi2

    a_correct = preds_a == labels
    b_correct = preds_b == labels
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    if (b + c) == 0:
        return b, c, 1.0
    if (b + c) < 25:
        # Exact binomial
        k = min(b, c)
        p = float(2 * binom.cdf(k, b + c, 0.5))
        p = min(1.0, p)
    else:
        chi_sq = (abs(b - c) - 1) ** 2 / (b + c)
        p = float(1 - chi2.cdf(chi_sq, df=1))
    return b, c, p


def paired_permutation(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Two-sided paired permutation test on any metric.

    Returns (metric_a, metric_b, p_value).
    """
    rng = np.random.default_rng(seed)
    m_a = float(metric_fn(labels, preds_a))
    m_b = float(metric_fn(labels, preds_b))
    observed = m_a - m_b

    count = 0
    N = len(labels)
    pa = preds_a.copy()
    pb = preds_b.copy()
    for _ in range(n):
        swap = rng.random(N) < 0.5
        # Swap per-sample
        new_a = np.where(swap, pb, pa)
        new_b = np.where(swap, pa, pb)
        diff = metric_fn(labels, new_a) - metric_fn(labels, new_b)
        if abs(diff) >= abs(observed):
            count += 1
    p = (count + 1) / (n + 1)
    return m_a, m_b, p


# Convenience metric functions to pair with the bootstrap.

def acc_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def macro_f1_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def bal_acc_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


def mcc_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(y_true, y_pred))


def kappa_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(y_true, y_pred))
