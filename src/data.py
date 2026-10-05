from sqlite3 import connect

import pandas as pd
import yfinance as yf

CONN = connect(":memory:")


def download_prices(tickers, start_date="2010-01-01"):
    """
    Download historical price data for the given tickers
    """
    data = yf.download(tickers, start=start_date, auto_adjust=True)[
        "Close"
    ]
    print(data.columns)
    return data


def clean_prices(data):
    """
    drop rows where every ticker is missing. No forward fill: stale prices give
    zero returns and understate volatility.
    """
    rs = data.shape[0]
    data = data.dropna(how="all")
    rs_clean = data.shape[0]
    print(f"dropped: {rs - rs_clean} rows with all missing values")
    return data


def to_returns(data):
    """
    convert prices to returns
    """
    returns = data.pct_change(fill_method=None).dropna()
    return returns


def save_prices(data, path):
    """
    save the price data to a table in the in-memory sqlite db
    """
    data.to_sql(path, con=CONN, if_exists="replace", index_label="Date")


def load_prices(path):
    """
    load price data saved by save_prices, with the dates as the index
    """
    return pd.read_sql(
        f'SELECT * FROM "{path}"', con=CONN, index_col="Date", parse_dates=["Date"]
    )
