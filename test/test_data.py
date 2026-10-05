import numpy as np
import pandas as pd
import pytest

from data import clean_prices, download_prices, load_prices, save_prices, to_returns


@pytest.mark.network
def test_download_prices():
    tickers = ["AAPL", "MSFT"]
    data = download_prices(tickers)
    assert not data.empty
    assert all(ticker in data.columns for ticker in tickers)


def test_clean_prices_drops_all_nan_rows():
    idx = pd.date_range("2024-01-01", periods=4)
    data = pd.DataFrame(
        {"AAPL": [1.0, np.nan, 3.0, 4.0], "MSFT": [1.0, np.nan, 3.0, np.nan]},
        index=idx,
    )
    cleaned = clean_prices(data)
    assert cleaned.shape == (3, 2)
    assert idx[1] not in cleaned.index
    # partial gaps are kept as NaN, not forward-filled
    assert np.isnan(cleaned.loc[idx[3], "MSFT"])


def test_save_and_load_prices_round_trip():
    tickers = ["AAPL", "MSFT"]
    data = pd.DataFrame(
        {ticker: [1.0, 2.0, 3.0] for ticker in tickers},
        index=pd.date_range("2024-01-01", periods=3, name="Date"),
    )
    save_prices(data, "test_prices")
    loaded = load_prices("test_prices")
    # index freq doesn't survive SQL, so don't compare it
    pd.testing.assert_frame_equal(data, loaded, check_freq=False)


def test_to_returns():
    prices = pd.DataFrame(
        {"AAPL": [100.0, 110.0, 99.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    returns = to_returns(prices)
    assert list(returns.index) == list(prices.index[1:])
    assert returns["AAPL"].values == pytest.approx([0.10, -0.10])


def test_to_returns_does_not_fill_gaps():
    idx = pd.date_range("2024-01-01", periods=4)
    prices = pd.DataFrame(
        {"AAPL": [100.0, 110.0, 99.0, 104.0], "MSFT": [100.0, np.nan, 100.0, 105.0]},
        index=idx,
    )
    returns = to_returns(prices)
    # the gap day and the day after have no MSFT return, so both rows are dropped
    # rather than producing a stale zero return
    assert list(returns.index) == [idx[3]]
    assert returns.loc[idx[3], "MSFT"] == pytest.approx(0.05)
