import numpy as np
import pytest

import optimise_prices as op


def make_market(elasticities, cost=0.5, lower=0.6, upper=1.4):
    n = len(elasticities)
    return op.Market(brands=[f"b{i}" for i in range(n)], elasticities=np.array(elasticities, dtype=float),
                     base_price=np.ones(n), base_units=np.full(n, 100.0), unit_cost=np.full(n, cost),
                     lower=np.full(n, lower), upper=np.full(n, upper))


def test_units_follow_constant_elasticity():
    market = make_market([[-2.0, 0.5], [0.5, -2.0]])
    u = op.units(market, np.array([1.1, 1.0]))
    assert np.allclose(u, [100 * 1.1 ** -2.0, 100 * 1.1 ** 0.5])


def test_single_product_optimum_matches_lerner_rule():
    """With elasticity e, the profit-maximising price is cost * e / (1 + e)."""
    market = make_market([[-3.0]])
    best = op.best_prices(market, ["b0"], steps=8001)
    assert best[0] == pytest.approx(0.5 * 3 / 2, abs=1e-3)


def test_optimum_stays_inside_bounds():
    market = make_market([[-1.1]], upper=1.2)  # inelastic: unbounded optimum would be very high
    assert op.best_prices(market, ["b0"], steps=101)[0] == pytest.approx(1.2)


def test_held_brands_keep_baseline_price():
    market = make_market([[-3.0, 0.3], [0.3, -3.0]])
    best = op.best_prices(market, ["b1"], steps=51)
    assert best[0] == 1.0


def test_elasticity_draws_centre_on_estimates():
    results = {
        "brands": ["a", "b"],
        "elasticities": {"a": {"a": -2.0, "b": 0.4}, "b": {"a": 0.3, "b": -1.5}},
        "covariance": {i: {j: {k: (0.01 if j == k else 0.0) for k in "ab"} for j in "ab"} for i in "ab"},
    }
    draws = op.draw_elasticities(results, draws=4000)
    assert draws.shape == (4000, 2, 2)
    assert np.allclose(draws.mean(axis=0), [[-2.0, 0.4], [0.3, -1.5]], atol=0.01)
