# podrisk

Portfolio risk analytics in Python: download prices, turn them into returns, and estimate Value at Risk (VaR) and Expected Shortfall (ES).

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Layout

| File | What it does |
| --- | --- |
| `src/data.py` | Downloads prices from Yahoo Finance, cleans them, converts them to returns, and saves/loads them in an in-memory SQLite database |
| `src/model.py` | Risk models: historical simulation VaR/ES, EWMA volatility, and parametric (normal or Student-t) VaR/ES |
| `src/book.py` | Example portfolio of dollar positions, grouped by desk |

## Models

| Function | Description |
| --- | --- |
| `hs_var`, `hs_es` | Historical simulation VaR and ES over a rolling window (500 days by default) |
| `ewma_vol` | EWMA volatility, λ = 0.94 by default |
| `param_var`, `param_es` | Parametric VaR and ES from a volatility, using a normal or Student-t distribution |

Every rolling estimate is shifted by one day, so the figure for a given date uses only data from earlier dates. Losses are returned as positive numbers.

## Example

```python
from data import clean_prices, download_prices, to_returns
from model import ewma_vol, hs_var, param_es, param_var

prices = clean_prices(download_prices(["SPY"]))
returns = to_returns(prices)["SPY"]

var_hs = hs_var(returns, alpha=0.01)

sigma = ewma_vol(returns)
var_t = param_var(sigma, alpha=0.01, dist="t", nu=5)
es_t = param_es(sigma, alpha=0.01, dist="t", nu=5)
```

Run it from `src/` or add `src` to your `PYTHONPATH`.

## Tests

```bash
uv run pytest
```

To skip the tests that need the network:

```bash
uv run pytest -m "not network"
```
