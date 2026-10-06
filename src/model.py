import numpy as np
import pandas as pd
import scipy.stats as stats


def hs_var(returns, alpha=0.05, window=500, min_periods=450):
    """
    calc historical simulation value at risk for a given return series and confidence level
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    var = (
        -returns.rolling(window=window, min_periods=min_periods)
        .quantile(alpha)
        .shift(1)
    )
    return var


def _tail_mean(x, alpha):
    """
    helper func to calc the tail mean of a series
    """
    return x[x <= x.quantile(alpha)].mean() if len(x) > 0 else float("nan")


def hs_es(returns, alpha=0.05, window=500, min_periods=450):
    """
    calc historical simulation expected shortfall: mean loss beyond the alpha quantile
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    es = (
        -returns.rolling(window=window, min_periods=min_periods)
        .apply(_tail_mean, raw=False, kwargs={"alpha": alpha})
        .shift(1)
    )
    return es


def ewma_vol(pnl, lam=0.94, min_periods=30) -> pd.Series:
    """
    calculate exponentially weighted moving average volatility of a PnL series
    """
    if not isinstance(pnl, pd.Series):
        pnl = pd.Series(pnl)

    squared_returns = pnl**2
    ewma_var = squared_returns.ewm(alpha=1 - lam, min_periods=min_periods).mean()
    return np.sqrt(ewma_var).shift(1)


def _check_dist(dist, nu):
    """
    helper func to validate the distribution arguments of the parametric models
    """
    if dist not in ("normal", "t"):
        raise ValueError("unsupported distribution. Use 'normal' or 't'")
    if dist == "t" and nu <= 2:
        raise ValueError(
            "nu must be greater than 2 for the t distribution to have finite variance"
        )


def param_var(sigma, alpha=0.01, dist="normal", nu=5) -> pd.Series:
    """
    calc parametric value at risk for a given volatility and tail probability alpha.
    sigma is the standard deviation of returns, so the t quantile is rescaled to unit variance
    """
    _check_dist(dist, nu)
    if dist == "normal":
        return -stats.norm.ppf(alpha) * sigma
    return -stats.t.ppf(alpha, df=nu) * np.sqrt((nu - 2) / nu) * sigma


def param_es(sigma, alpha=0.01, dist="normal", nu=5) -> pd.Series:
    """
    calc parametric expected shortfall for a given volatility and tail probability alpha.
    sigma is the standard deviation of returns, so the t result is rescaled to unit variance
    """
    _check_dist(dist, nu)
    if dist == "normal":
        z_alpha = stats.norm.ppf(alpha)
        return sigma * stats.norm.pdf(z_alpha) / alpha
    z_alpha = stats.t.ppf(alpha, df=nu)
    pdf_alpha = stats.t.pdf(z_alpha, df=nu)
    es_unit_scale = (nu + z_alpha**2) / (nu - 1) * pdf_alpha / alpha
    return es_unit_scale * np.sqrt((nu - 2) / nu) * sigma
