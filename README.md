# podrisk

Value at Risk (VaR) and Expected Shortfall (ES) models in Python, with statistical backtests and historical stress tests, applied to a single equity index and to a hedged multi-desk portfolio.

The project asks a practical question: **can a VaR model's forecasts be trusted, and does the answer depend on the portfolio?** Each model forecasts tomorrow's 99% VaR using only past data. Each forecast is then compared with what actually happened, using standard statistical coverage tests.

## Key findings

- **On SPY, every model is rejected.** EWMA forgets crashes within weeks and breaches too often in calm markets. Historical simulation reacts too slowly, so its breaches cluster in crises. GARCH comes closest, but SPY's losses are left-skewed (big down days are larger than big up days), which a symmetric Student-t tail cannot capture.
- **On a hedged long/short book, the results reverse.** Hedging cancels most of the market's crash asymmetry, so the book's tail is close to normal. EWMA with a Student-t passes every test on every desk, while GARCH t becomes too conservative (20 breaches against 32 expected).
- **The same model can be too risky for one portfolio and too cautious for another.** A fixed tail parameter (ν = 5) is too thin for SPY and too fat for the book.
- **Diversification between desks reduces VaR by 28% on average.** The two desks' P&L has a −0.10 correlation, so the book's VaR is well below the sum of the desk VaRs.
- **ES holds up on the book.** Every model passes the Acerbi–Székely test at 97.5%.
- **2022 is the stress scenario that matters.** The rates/FX desk loses $1.7m, and gains on the equity desk cut the book's loss to $0.7m.

## Results

All results are one-day 99% VaR, so a correct model breaches on about 1% of days. Each test reports a p-value, and a p-value below 0.05 rejects the model.

### SPY

Daily SPY returns, 2013-12-24 to 2026-10-16 (3,222 days, 32.2 breaches expected).

| Model | Breaches | Kupiec p | Christoffersen p | Cond. coverage p |
| --- | ---: | ---: | ---: | ---: |
| Historical simulation | 49 | 0.006 | <0.001 | <0.001 |
| EWMA, normal | 75 | <0.001 | 0.009 | <0.001 |
| EWMA, Student-t | 58 | <0.001 | 0.004 | <0.001 |
| GARCH(1,1), Student-t | **46** | 0.021 | 0.004 | 0.001 |

![SPY daily returns against each model's VaR, with breaches marked](docs/img/spy_var_backtest.png)

### Hedged book

Hypothetical daily P&L of the example book in `src/book.py`, 2013-12-30 to 2026-10-16 (3,215 days, 32.2 breaches expected). The book has two desks:

- **equity_ls:** long $20m across AAPL, MSFT, JPM and UNH, short $20m SPY
- **rates_fx:** long TLT against short IEF ($10m each side), and long GBP against short EUR ($5m each side)

Results for the whole book:

| Model | Breaches | Kupiec p | Christoffersen p | Cond. coverage p |
| --- | ---: | ---: | ---: | ---: |
| Historical simulation | 42 | 0.093 | 0.291 | 0.140 |
| EWMA, normal | 36 | 0.495 | 0.366 | 0.527 |
| EWMA, Student-t | 25 | 0.191 | 0.531 | 0.350 |
| GARCH(1,1), Student-t | 20 | 0.021 | 0.616 | 0.062 |

EWMA with a Student-t passes every test on both desks and on the whole book. The notebook has the per-desk results.

![Book VaR against the sum of the two desk VaRs](docs/img/book_diversification.png)

| Mean 99% VaR (GARCH t) | |
| --- | ---: |
| equity_ls | $292k |
| rates_fx | $148k |
| Sum of desks | $440k |
| **Book** | **$313k** |

### Expected shortfall

97.5% ES on the whole book, same dates (80.4 breaches expected). `tail_ratio` is the average loss on breach days divided by the average predicted ES. Near 1 is good.

| Model | Breaches | tail_ratio | Z2 | p |
| --- | ---: | ---: | ---: | ---: |
| Historical simulation | 90 | 1.06 | −0.20 | 0.072 |
| EWMA, normal | 72 | 1.15 | −0.02 | 0.428 |
| EWMA, Student-t | 66 | 1.02 | 0.18 | 0.951 |
| GARCH(1,1), Student-t | 62 | 1.02 | 0.21 | 0.974 |

All four pass on the book. At desk level, HS fails on equity_ls and EWMA normal fails on rates_fx.

## Stress test

Each crisis's actual daily returns, replayed on today's book with no trading. Whole book, $ P&L:

| Scenario | Worst day | Worst 10 days | Total |
| --- | ---: | ---: | ---: |
| GFC 2008 | −$1.33m | −$1.74m | +$0.41m |
| COVID 2020 | −$0.82m | −$0.20m | +$0.57m |
| Rates 2022 | −$0.37m | −$0.74m | −$0.71m |

The hedged equity desk makes money over each crisis. In 2022 the rates/FX desk (long TLT, short IEF) loses $1.7m as long rates rise.

## Methodology

**Models.** Every model produces a one-day-ahead forecast using only data up to the previous day, and tests check this.

| Model | Volatility | Tail |
| --- | --- | --- |
| Historical simulation | None: uses the empirical 1% quantile of the last 500 days | Empirical |
| EWMA | Exponentially weighted, λ = 0.94 (RiskMetrics) | Normal or Student-t (ν = 5) |
| GARCH(1,1) | Refitted every 21 days on the trailing 1,000 days using [`arch`](https://github.com/bashtage/arch) | Student-t (ν = 5) |

The Student-t is rescaled to unit variance, so the volatility forecast keeps its meaning and only the shape of the tail changes.

**Backtests.**

| Test | Question | Distribution |
| --- | --- | --- |
| Kupiec (1995) | Is the breach rate equal to 1%? Too few breaches also fails. | χ², 1 df |
| Christoffersen (1998) | Are breaches independent, or does one breach make another more likely the next day? | χ², 1 df |
| Conditional coverage | Both at once | χ², 2 df |
| Acerbi–Székely Z2 (2014) | Does ES match the average loss on breach days? | Normal approximation, one-sided |

**Stress test.** Daily returns from three windows (GFC 2008, COVID 2020, Rates 2022) are applied to the current positions.

**Portfolio P&L.** The book's daily P&L is the sum of position × return. Positions are held at fixed dollar amounts, so this is hypothetical P&L. Only days when every market traded are kept: FX trades on US holidays, and the FX return on the next shared day covers the gap.

## Limitations and next steps

- **ν is fixed at 5.** The results show it should be estimated for each series. Next step: use the ν fitted by GARCH, and try a skewed-t for SPY.
- **The stress test only covers three past crises**, with positions held fixed. A new kind of shock won't show up.
- **The book is modelled as a single P&L series.** A position-level model would allow component VaR, showing which positions drive the risk.
- **Data is downloaded live.** Results shift slightly each time the notebooks are rerun.

## Project structure

| Path | Contents |
| --- | --- |
| `src/data.py` | Price download (Yahoo Finance), cleaning, and conversion to returns |
| `src/model.py` | VaR/ES models: historical simulation, EWMA, GARCH, parametric normal and Student-t |
| `src/backtesting.py` | Breaches, the Kupiec, Christoffersen and conditional coverage tests, and the Z2 ES test |
| `src/stresstesting.py` | Historical scenario stress test |
| `src/book.py` | Example portfolio of dollar positions by desk |
| `notebooks/backtesting.ipynb` | SPY backtest of all four models |
| `notebooks/book_backtest.ipynb` | Desk and book backtests, ES backtest, and diversification |
| `notebooks/stress_test.ipynb` | Stress test of the book |
| `test/` | Unit tests, including checks that no model uses future data |

## Usage

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```python
from data import clean_prices, download_prices, to_returns
from model import garch_vol, param_var
from backtesting import conditional_coverage, exceedance, kupiec

returns = to_returns(clean_prices(download_prices(["SPY"])))["SPY"]

sigma = garch_vol(returns)
var = param_var(sigma, alpha=0.01, dist="t", nu=5)

hits = exceedance(returns, var)
lr, p = kupiec(hits, alpha=0.01)
lr, p = conditional_coverage(hits, alpha=0.01)
```

Run from `src/`, or add `src` to your `PYTHONPATH`.

## Tests

```bash
uv run pytest                    # all tests
uv run pytest -m "not network"   # skip tests that download data
```
