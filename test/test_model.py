import numpy as np
import pandas as pd
import pytest

from model import hs_es, hs_var


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0, 0.01, 2000))


def test_hs_var_and_es_small_example():
    returns = pd.Series([0.03, -0.05, 0.02, -0.01, 0.0, 0.01])
    var = hs_var(returns, alpha=0.2, window=5, min_periods=5)
    es = hs_es(returns, alpha=0.2, window=5, min_periods=5)
    # 20% quantile of the first five returns, linearly interpolated: -0.05 + 0.8 * 0.04
    assert var.iloc[5] == pytest.approx(0.018)
    # only -0.05 is at or below that quantile
    assert es.iloc[5] == pytest.approx(0.05)


def test_hs_var_and_es_match_normal_theory(normal_returns):
    # for N(0, 0.01): 95% VaR = 1.645 * 0.01, ES = 0.01 * pdf(1.645) / 0.05
    assert hs_var(normal_returns).mean() == pytest.approx(0.01645, rel=0.1)
    assert hs_es(normal_returns).mean() == pytest.approx(0.0206, rel=0.1)


def test_hs_es_at_least_hs_var(normal_returns):
    var = hs_var(normal_returns)
    es = hs_es(normal_returns)
    both = var.notna() & es.notna()
    assert both.any()
    assert (es[both] >= var[both]).all()


@pytest.mark.parametrize("func", [hs_var, hs_es])
def test_no_look_ahead(func, normal_returns):
    # a shock on day t must not affect the estimate for day t
    shocked = normal_returns.copy()
    shocked.iloc[1000] = -0.5
    before = func(normal_returns)
    after = func(shocked)
    assert after.iloc[1000] == pytest.approx(before.iloc[1000])
    assert after.iloc[1001] > before.iloc[1001]


@pytest.mark.parametrize("func", [hs_var, hs_es])
def test_warm_up_and_single_gap(func, normal_returns):
    returns = normal_returns.copy()
    returns.iloc[700] = np.nan
    result = func(returns, window=500, min_periods=450)
    # warm-up: min_periods observations plus the one-day shift
    assert result.iloc[:450].isna().all()
    # one missing return does not blank out the following window
    assert result.iloc[450:].notna().all()


@pytest.mark.parametrize("func", [hs_var, hs_es])
def test_accepts_numpy_array(func, normal_returns):
    result = func(normal_returns.to_numpy())
    assert isinstance(result, pd.Series)
    pd.testing.assert_series_equal(result, func(normal_returns))
