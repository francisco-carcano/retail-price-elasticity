"""Profit-maximising prices for the tuna category, from the estimated elasticities.

Demand around the baseline store-week follows the fitted log-log model:
    units_i(p) = U_i * prod_j (p_j / P_j) ** e_ij
The retailer sets every shelf price, so the objective is profit across the whole category,
sum_i (p_i - c_i) * units_i. Prices are searched on a grid inside the range seen in the data,
because the constant-elasticity curve is not reliable outside it. Uncertainty comes from
redrawing the elasticities from their estimated sampling distribution.

Usage: python src/optimise_prices.py   (writes results/pricing.md)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from demand_model import ROOT, load_results

REPORT_PATH = ROOT / "results" / "pricing.md"
PRIVATE_LABEL = "Dominick's"


@dataclass(frozen=True)
class Market:
    brands: list[str]
    elasticities: np.ndarray  # [demand brand, price brand]
    base_price: np.ndarray
    base_units: np.ndarray
    unit_cost: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_results(cls, results: dict) -> "Market":
        brands = results["brands"]
        base = results["baseline"]
        col = lambda key: np.array([base[b][key] for b in brands])  # noqa: E731
        bounds = col("price_bounds")
        return cls(brands=brands,
                   elasticities=np.array([[results["elasticities"][i][j] for j in brands] for i in brands]),
                   base_price=col("price"), base_units=col("units"), unit_cost=col("unit_cost"),
                   lower=bounds[:, 0], upper=bounds[:, 1])

    def index(self, brand: str) -> int:
        return self.brands.index(brand)


def units(market: Market, prices: np.ndarray, elasticities: np.ndarray | None = None) -> np.ndarray:
    """Units per brand for each row of `prices` (shape [..., brands])."""
    e = market.elasticities if elasticities is None else elasticities
    log_ratio = np.log(prices / market.base_price)
    return market.base_units * np.exp(log_ratio @ e.T)


def category_profit(market: Market, prices: np.ndarray, elasticities: np.ndarray | None = None) -> np.ndarray:
    return ((prices - market.unit_cost) * units(market, prices, elasticities)).sum(axis=-1)


def price_grid(market: Market, movable: list[str], steps: int) -> np.ndarray:
    """Every combination of prices for the `movable` brands; the others stay at baseline."""
    axes = [np.linspace(market.lower[i], market.upper[i], steps) if b in movable else [market.base_price[i]]
            for i, b in enumerate(market.brands)]
    return np.array(list(itertools.product(*axes)))


def best_prices(market: Market, movable: list[str], steps: int, elasticities: np.ndarray | None = None) -> np.ndarray:
    grid = price_grid(market, movable, steps)
    return grid[np.argmax(category_profit(market, grid, elasticities))]


def draw_elasticities(results: dict, draws: int, seed: int = 0) -> np.ndarray:
    """Samples each brand's row of elasticities from a normal with its estimated covariance."""
    rng = np.random.default_rng(seed)
    brands = results["brands"]
    rows = []
    for i in brands:
        mean = [results["elasticities"][i][j] for j in brands]
        cov = [[results["covariance"][i][j][k] for k in brands] for j in brands]
        rows.append(rng.multivariate_normal(mean, cov, size=draws))
    return np.stack(rows, axis=1)  # [draw, demand brand, price brand]


@dataclass(frozen=True)
class Recommendation:
    movable: list[str]
    prices: np.ndarray
    profit_change: float  # relative to baseline, point estimate
    profit_change_interval: tuple[float, float]  # 90% interval over elasticity draws
    price_interval: dict[str, tuple[float, float]]


def recommend(market: Market, results: dict, movable: list[str], steps: int, draws: int) -> Recommendation:
    """Optimal prices and their uncertainty. For each elasticity draw the prices are re-optimised
    and the profit change is measured under that same draw."""
    prices = best_prices(market, movable, steps)
    base_profit = category_profit(market, market.base_price)
    sampled_prices, sampled_change = [], []
    for e in draw_elasticities(results, draws):
        p = best_prices(market, movable, steps, e)
        sampled_prices.append(p)
        sampled_change.append(category_profit(market, p, e) / category_profit(market, market.base_price, e) - 1)
    sampled_prices = np.array(sampled_prices)
    return Recommendation(
        movable=movable,
        prices=prices,
        profit_change=float(category_profit(market, prices) / base_profit - 1),
        profit_change_interval=tuple(np.quantile(sampled_change, [0.05, 0.95])),
        price_interval={b: tuple(np.quantile(sampled_prices[:, market.index(b)], [0.05, 0.95])) for b in movable},
    )


def report(market: Market, recommendations: list[tuple[str, Recommendation]]) -> str:
    lines = ["# Pricing results (generated by `src/optimise_prices.py`)", "",
             "Average store-week. Category = the four chunk light 6 oz brands. Prices limited to the 5th-95th "
             "percentile of prices seen in the data. Intervals: 90%, from redrawing the elasticities.", "",
             "## Baseline", "", "| Brand | Price | Unit cost | Margin | Units |", "| --- | --- | --- | --- | --- |"]
    for i, b in enumerate(market.brands):
        p, c = market.base_price[i], market.unit_cost[i]
        lines.append(f"| {b} | ${p:.2f} | ${c:.2f} | {(p - c) / p:.0%} | {market.base_units[i]:.0f} |")
    for title, r in recommendations:
        new_units = units(market, r.prices)
        lines += ["", f"## {title}", "",
                  f"Category profit change: **{r.profit_change:+.1%}** "
                  f"(90% interval {r.profit_change_interval[0]:+.1%} to {r.profit_change_interval[1]:+.1%}).", "",
                  "| Brand | Baseline price | Recommended price | Price interval | Units change |",
                  "| --- | --- | --- | --- | --- |"]
        for i, b in enumerate(market.brands):
            interval = (f"${r.price_interval[b][0]:.2f}-${r.price_interval[b][1]:.2f}" if b in r.movable else "held")
            corner = " (top of tested range)" if b in r.movable and np.isclose(r.prices[i], market.upper[i]) else ""
            lines.append(f"| {b} | ${market.base_price[i]:.2f} | ${r.prices[i]:.2f}{corner} | {interval} | "
                         f"{new_units[i] / market.base_units[i] - 1:+.0%} |")
    return "\n".join(lines) + "\n"


def price_sweep(market: Market, results: dict, brand: str, prices: list[float], draws: int) -> str:
    """Category profit change as one brand's price moves, others held, with 90% intervals."""
    i = market.index(brand)
    sampled = draw_elasticities(results, draws)
    lines = ["", f"## Category profit as the {brand} price moves (others held)", "",
             "| Price | Change vs baseline | 90% interval | Its own units |", "| --- | --- | --- | --- |"]
    for p in prices:
        x = market.base_price.copy()
        x[i] = p
        change = category_profit(market, x) / category_profit(market, market.base_price) - 1
        draws_change = [category_profit(market, x, e) / category_profit(market, market.base_price, e) - 1
                        for e in sampled]
        low, high = np.quantile(draws_change, [0.05, 0.95])
        own = units(market, x)[i] / market.base_units[i] - 1
        lines.append(f"| ${p:.2f} | {change:+.1%} | {low:+.1%} to {high:+.1%} | {own:+.0%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    results = load_results()
    market = Market.from_results(results)
    recommendations = [
        ("Move only the private label price", recommend(market, results, [PRIVATE_LABEL], steps=141, draws=1000)),
        ("Move all four prices", recommend(market, results, market.brands, steps=15, draws=200)),
    ]
    sweep = price_sweep(market, results, PRIVATE_LABEL, [0.69, 0.75, 0.79, 0.85, 0.89, 0.94], draws=1000)
    REPORT_PATH.write_text(report(market, recommendations) + sweep, encoding="utf-8")
    print(REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
