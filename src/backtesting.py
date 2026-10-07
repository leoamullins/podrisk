import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import xlogy


def exceedance(returns, var):
    """
    calculate the exceedance ratio of a return series given a VaR serues
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    if not isinstance(var, pd.Series):
        var = pd.Series(var)

    df = pd.concat({"ret": returns, "var": var}, axis=1).dropna()
    return -df["ret"] > df["var"]


def summary(hits, alpha):
    # give exceedance summary statistics
    n, x = len(hits), int(hits.sum())
    return {"n": n, "exceedances": x, "expected": n * alpha, "hit_rate": x / n}


def kupiec(hits, alpha):
    """
    Kupiec unconditional coverage test. small p-value means the hit rate is wrong
    """
    n, x = len(hits), int(hits.sum())

    # building the LR_pof
    p_hat = x / n
    ll_null = xlogy(n - x, 1 - alpha) + xlogy(x, alpha)
    ll_alt = xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = -2 * (ll_null - ll_alt)
    return lr, stats.chi2.sf(lr, 1)


def christoffersen(hits, alpha):
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]

    # count day to day transitions
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))

    # breach probabilities
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (
        xlogy(n00, 1 - pi01)
        + xlogy(n01, pi01)
        + xlogy(n10, 1 - pi11)
        + xlogy(n11, pi11)
    )
    lr = -2 * (ll_null - ll_alt)
    return lr, stats.chi2.sf(lr, 1)


def conditional_coverage(hits, alpha):
    lr = kupiec(hits, alpha)[0] + christoffersen(hits, alpha)[0]
    return lr, stats.chi2.sf(lr, 2)
