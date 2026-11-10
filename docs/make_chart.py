"""Draws the README chart from the saved model results. Usage: python docs/make_chart.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import optimise_prices as op  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e1e0d9", base="#c3c2b7",
                  series="#2a78d6", muted="#c3c2b7"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#2c2c2a", base="#383835",
                 series="#3987e5", muted="#5a5955"),
}
OUT_DIR = Path(__file__).resolve().parent / "figures"


def style(ax, t, grid_axis="y"):
    ax.set_facecolor(t["surface"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def profit_curve(market, results, prices, draws=500):
    i = market.index(op.PRIVATE_LABEL)
    sampled = op.draw_elasticities(results, draws)
    base, base_draws = op.category_profit(market, market.base_price), [
        op.category_profit(market, market.base_price, e) for e in sampled]
    mid, low, high = [], [], []
    for p in prices:
        x = market.base_price.copy()
        x[i] = p
        mid.append(op.category_profit(market, x) / base - 1)
        changes = [op.category_profit(market, x, e) / b - 1 for e, b in zip(sampled, base_draws)]
        low.append(np.quantile(changes, 0.05))
        high.append(np.quantile(changes, 0.95))
    return np.array(mid) * 100, np.array(low) * 100, np.array(high) * 100


def draw(name, market, results):
    t = THEMES[name]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.9), facecolor=t["surface"],
                                      gridspec_kw={"width_ratios": [1, 1.25]})

    own = np.diag(market.elasticities)
    labels = [b if b != op.PRIVATE_LABEL else "Dominick's\n(store brand)" for b in market.brands]
    labels = [label.replace("Chicken of the Sea", "Chicken of\nthe Sea") for label in labels]
    colors = [t["series"] if b == op.PRIVATE_LABEL else t["muted"] for b in market.brands]
    y = np.arange(len(own))[::-1]
    left.barh(y, own, color=colors, height=0.55)
    for yi, v in zip(y, own):
        left.text(v - 0.08, yi, f"{v:.2f}", va="center", ha="right", color=t["ink"], fontsize=10)
    left.set_yticks(y, labels)
    left.set_xlim(-3.9, 0)
    left.axvline(0, color=t["base"], linewidth=1)
    left.set_title("Units change for a 1% price rise (%)", color=t["ink"], fontsize=11, loc="left")
    style(left, t, grid_axis="x")
    left.grid(False)
    left.set_xticks([])
    left.tick_params(axis="y", colors=t["ink"], labelsize=10)

    i = market.index(op.PRIVATE_LABEL)
    prices = np.linspace(market.lower[i], market.upper[i], 60)
    mid, low, high = profit_curve(market, results, prices)
    right.fill_between(prices, low, high, color=t["series"], alpha=0.18, linewidth=0)
    right.plot(prices, mid, color=t["series"], linewidth=2)
    right.axhline(0, color=t["base"], linewidth=1)
    base_price = market.base_price[i]
    right.axvline(base_price, color=t["base"], linewidth=1, linestyle="--")
    right.text(base_price + 0.005, mid.min(), f"Typical\n${base_price:.2f}", color=t["ink2"], fontsize=9, va="bottom")
    right.plot([prices[-1]], [mid[-1]], "o", color=t["series"], markersize=8, markeredgecolor=t["surface"], markeredgewidth=2)
    right.annotate(f"+{mid[-1]:.1f}% at ${prices[-1]:.2f},\nhighest price in the data", (prices[-1], mid[-1]),
                   xytext=(-4, -100), textcoords="offset points", ha="right", color=t["ink"], fontsize=10)
    right.xaxis.set_major_formatter(lambda v, _: f"${v:.2f}")
    right.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
    right.set_title("Category profit vs store-brand price, 90% interval", color=t["ink"], fontsize=11, loc="left")
    style(right, t)

    fig.suptitle("Store-brand shoppers barely react to its price, so raising it pays",
                 color=t["ink"], fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"pricing_{name}.png", dpi=160, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    results = op.load_results()
    market = op.Market.from_results(results)
    for theme in THEMES:
        draw(theme, market, results)
