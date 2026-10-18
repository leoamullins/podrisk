import pandas as pd

SCENARIOS = {
    "GFC 2008": ("2008-09-15", "2008-11-20"),
    "COVID 2020": ("2020-02-20", "2020-03-23"),
    "Rates 2022": ("2022-01-03", "2022-10-14"),
}


def stress(returns, book, scenarios=SCENARIOS):
    rows = []
    for name, (start, end) in scenarios.items():
        r = returns.loc[start:end]
        desk_pnl = {
            desk: (r[list(pos)] * pd.Series(pos)).sum(axis=1, min_count=1)
            for desk, pos in book.items()
        }
        desk_pnl["BOOK"] = pd.concat(desk_pnl, axis=1).sum(axis=1, min_count=1)
        for desk, pnl in desk_pnl.items():
            rows.append(
                {
                    "scenario": name,
                    "desk": desk,
                    "worst_1d": pnl.min(),
                    "worst_10d": pnl.dropna().rolling(10).sum().min(),
                    "total": pnl.sum(),
                }
            )
    return pd.DataFrame(rows)
