import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from backtesting import christoffersen, conditional_coverage, exceedance, kupiec, summary


@pytest.fixture
def correct_hits():
    # hits from a model whose true breach rate really is 1%
    rng = np.random.default_rng(0)
    return pd.Series(rng.random(5000) < 0.01)


def test_exceedance_sign_convention():
    returns = pd.Series([-0.03, -0.01, 0.02, -0.02])
    var = pd.Series([0.02, 0.02, 0.02, 0.02])
    # a loss bigger than VaR is a breach, a smaller loss, a gain or an exact hit is not
    assert exceedance(returns, var).tolist() == [True, False, False, False]


def test_exceedance_drops_warm_up():
    returns = pd.Series([-0.05, -0.05, -0.05, 0.01])
    var = pd.Series([np.nan, np.nan, 0.02, 0.02])
    hits = exceedance(returns, var)
    assert hits.index.tolist() == [2, 3]
    assert hits.tolist() == [True, False]


def test_exceedance_aligns_on_index():
    # misaligned inputs are matched by date, not by position
    dates = pd.date_range("2024-01-01", periods=3)
    returns = pd.Series([-0.05, 0.01, -0.05], index=dates)
    var = pd.Series([0.02, 0.02], index=dates[1:])
    assert exceedance(returns, var).tolist() == [False, True]


def test_summary():
    hits = pd.Series([True, False, False, False, True] + [False] * 95)
    assert summary(hits, alpha=0.01) == {
        "n": 100,
        "exceedances": 2,
        "expected": pytest.approx(1.0),
        "hit_rate": pytest.approx(0.02),
    }


def test_kupiec_zero_when_hit_rate_equals_alpha():
    hits = pd.Series([True] * 10 + [False] * 990)
    lr, p_value = kupiec(hits, alpha=0.01)
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p_value == pytest.approx(1.0)


def test_kupiec_worked_example():
    # 18 breaches in 1000 days against 1% VaR
    hits = pd.Series([True] * 18 + [False] * 982)
    lr, p_value = kupiec(hits, alpha=0.01)
    assert lr == pytest.approx(5.22, abs=0.01)
    assert p_value == pytest.approx(0.022, abs=0.001)


def test_kupiec_passes_correct_model(correct_hits):
    _, p_value = kupiec(correct_hits, alpha=0.01)
    assert p_value > 0.05


def test_kupiec_rejects_wrong_alpha(correct_hits):
    # the same hits judged against a 5% model are far too few
    _, p_value = kupiec(correct_hits, alpha=0.05)
    assert p_value < 1e-10


def test_kupiec_rejects_too_few_breaches():
    hits = pd.Series([True] * 3 + [False] * 997)
    _, p_value = kupiec(hits, alpha=0.01)
    assert p_value < 0.05


def test_kupiec_no_breaches_is_finite():
    hits = pd.Series([False] * 1000)
    lr, p_value = kupiec(hits, alpha=0.01)
    assert np.isfinite(lr)
    assert np.isfinite(p_value)
    # 0 breaches when 10 are expected should fail
    assert p_value < 0.05


def _hits_at(positions, n=1000):
    hits = pd.Series(False, index=range(n))
    hits.iloc[positions] = True
    return hits


def test_christoffersen_spread_out_hits_pass():
    # 10 breaches, never on consecutive days
    hits = _hits_at(range(50, 1000, 100))
    lr, p_value = christoffersen(hits, alpha=0.01)
    assert lr == pytest.approx(0.20, abs=0.01)
    assert p_value == pytest.approx(0.65, abs=0.01)


def test_christoffersen_clustered_hits_fail():
    # the same 10 breaches, but as 5 back-to-back pairs
    hits = _hits_at([i + d for i in range(50, 1000, 200) for d in (0, 1)])
    lr, p_value = christoffersen(hits, alpha=0.01)
    assert lr == pytest.approx(35.27, abs=0.01)
    assert p_value < 1e-8


def test_christoffersen_passes_independent_hits(correct_hits):
    _, p_value = christoffersen(correct_hits, alpha=0.01)
    assert p_value > 0.05


def test_christoffersen_rejects_too_regular_hits():
    # breaches exactly every 20 days never follow each other: that is dependence too
    _, p_value = christoffersen(_hits_at(range(10, 1000, 20)), alpha=0.01)
    assert p_value < 0.05


@pytest.mark.parametrize("positions", [[], [999], [0]])
def test_christoffersen_no_or_one_breach_is_finite(positions):
    # no breach, or no day after a breach: pi11 is undefined and must not give nan
    lr, p_value = christoffersen(_hits_at(positions), alpha=0.01)
    assert np.isfinite(lr)
    assert np.isfinite(p_value)


def test_conditional_coverage_is_sum_of_kupiec_and_christoffersen(correct_hits):
    lr, p_value = conditional_coverage(correct_hits, alpha=0.01)
    expected = kupiec(correct_hits, 0.01)[0] + christoffersen(correct_hits, 0.01)[0]
    assert lr == pytest.approx(expected)
    assert p_value == pytest.approx(stats.chi2.sf(expected, 2))


def test_conditional_coverage_catches_either_failure(correct_hits):
    # independent 1% hits judged against 0.2% VaR: no clustering, but 5x too many
    assert christoffersen(correct_hits, 0.002)[1] > 0.05
    assert conditional_coverage(correct_hits, 0.002)[1] < 0.05
    # 10 breaches in pairs: right count, but clustered
    clustered = _hits_at([i + d for i in range(50, 1000, 200) for d in (0, 1)])
    assert kupiec(clustered, 0.01)[1] > 0.05
    assert conditional_coverage(clustered, 0.01)[1] < 0.05
