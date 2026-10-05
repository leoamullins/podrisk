import pandas as pd


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
