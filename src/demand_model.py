"""Log-log demand model: own and cross price elasticities for the four tuna brands.

For each brand i, over store-weeks:
    log(units_i) = store effect + week effect + sum_j e_ij * log(price_j) [+ b * promo_share_i] + error
e_ii is the own-price elasticity and e_ij (j != i) the cross-price elasticities. Store effects
absorb store size and location; week effects absorb seasonality and chain-wide shocks such as
Lent. Standard errors are clustered by week, because prices are set chain-wide each week.

Usage: python src/demand_model.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from build_panel import BRANDS, PANEL_PATH, _slug


@dataclass(frozen=True)
class BrandFit:
    brand: str
    elasticities: pd.Series  # indexed by the brand whose price moves
    std_errors: pd.Series
    promo_effect: float | None
    r_squared: float
    holdout_r_squared: float
    n_obs: int


def price_terms() -> list[str]:
    return [f"np.log(price_{_slug(b)})" for b in BRANDS]


def formula(control_promo: bool) -> str:
    terms = price_terms() + (["promo_share"] if control_promo else [])
    return "np.log(units) ~ " + " + ".join(terms) + " + C(store) + C(week)"


def fit_brand(panel: pd.DataFrame, brand: str, control_promo: bool = True, holdout_share: float = 0.2) -> BrandFit:
    """Fits on the earlier weeks and reports R^2 on the last `holdout_share` of weeks too."""
    data = panel[panel["brand"] == brand]
    weeks = np.sort(data["week"].unique())
    cutoff = weeks[int(len(weeks) * (1 - holdout_share))]
    train, test = data[data["week"] < cutoff], data[data["week"] >= cutoff]

    full = smf.ols(formula(control_promo), data=data).fit(cov_type="cluster", cov_kwds={"groups": data["week"]})
    holdout = _holdout_r_squared(train, test, control_promo)
    names = dict(zip(price_terms(), BRANDS))
    return BrandFit(
        brand=brand,
        elasticities=full.params[list(names)].rename(index=names),
        std_errors=full.bse[list(names)].rename(index=names),
        promo_effect=float(full.params["promo_share"]) if control_promo else None,
        r_squared=float(full.rsquared),
        holdout_r_squared=holdout,
        n_obs=int(full.nobs),
    )


def _holdout_r_squared(train: pd.DataFrame, test: pd.DataFrame, control_promo: bool) -> float:
    """Out-of-time fit. Week effects are unknown for future weeks, so the holdout model swaps
    them for a smooth trend and week-of-year seasonality; it checks the price terms generalise."""
    f = formula(control_promo).replace("C(week)", "week + np.sin(2*np.pi*week/52) + np.cos(2*np.pi*week/52)")
    model = smf.ols(f, data=train).fit()
    test = test[test["store"].isin(train["store"].unique())]
    actual = np.log(test["units"])
    residual = actual - model.predict(test)
    return float(1 - (residual ** 2).sum() / ((actual - actual.mean()) ** 2).sum())


def elasticity_matrix(fits: list[BrandFit]) -> pd.DataFrame:
    """Rows: brand whose demand responds. Columns: brand whose price changes."""
    return pd.DataFrame({f.brand: f.elasticities for f in fits}).T.loc[list(BRANDS), list(BRANDS)]


def fit_all(panel: pd.DataFrame, control_promo: bool = True) -> list[BrandFit]:
    return [fit_brand(panel, b, control_promo) for b in BRANDS]


def summary(fits: list[BrandFit]) -> str:
    lines = ["Elasticity matrix (row demand, column price):", elasticity_matrix(fits).round(2).to_string(), ""]
    for f in fits:
        promo = "" if f.promo_effect is None else f", promo {f.promo_effect:+.2f}"
        lines.append(f"{f.brand}: own {f.elasticities[f.brand]:.2f} (se {f.std_errors[f.brand]:.2f}){promo}, "
                     f"R2 {f.r_squared:.2f}, holdout R2 {f.holdout_r_squared:.2f}, n {f.n_obs:,}")
    return "\n".join(lines)


if __name__ == "__main__":
    panel = pd.read_csv(PANEL_PATH)
    for control in (True, False):
        print(f"\n=== Promotion control: {control} ===")
        print(summary(fit_all(panel, control_promo=control)))
