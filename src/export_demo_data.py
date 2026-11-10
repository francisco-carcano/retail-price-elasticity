"""Writes the aggregated estimates the interactive page needs to docs/demo_data.json.

Only model outputs go out (elasticities, a baseline store-week, price ranges and elasticity
draws for the uncertainty band), never the raw Dominick's data.
Usage: python src/export_demo_data.py
"""
from __future__ import annotations

import json

import numpy as np

import optimise_prices as op
from demand_model import ROOT, load_results

DEMO_DATA_PATH = ROOT / "docs" / "demo_data.json"
DRAWS = 300


def demo_data(results: dict, draws: int = DRAWS) -> dict:
    market = op.Market.from_results(results)
    r = lambda x: np.round(x, 4).tolist()  # noqa: E731
    return {
        "brands": market.brands,
        "private_label": op.PRIVATE_LABEL,
        "elasticities": r(market.elasticities),
        "base_price": r(market.base_price),
        "base_units": r(market.base_units),
        "unit_cost": r(market.unit_cost),
        "lower": r(market.lower),
        "upper": r(market.upper),
        "draws": r(op.draw_elasticities(results, draws)),
    }


if __name__ == "__main__":
    DEMO_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_DATA_PATH.write_text(json.dumps(demo_data(load_results())), encoding="utf-8")
    print(f"Wrote {DEMO_DATA_PATH} ({DEMO_DATA_PATH.stat().st_size / 1024:.0f} KB)")
