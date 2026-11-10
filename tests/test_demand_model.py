import numpy as np
import pandas as pd
import pytest

import demand_model as dm
from build_panel import BRANDS, _slug

TRUE = pd.DataFrame(  # rows: demand, columns: price
    [[-2.5, 0.5, 0.3, 0.1],
     [0.6, -3.0, 0.2, 0.1],
     [0.4, 0.3, -2.0, 0.2],
     [0.2, 0.1, 0.1, -1.2]], index=list(BRANDS), columns=list(BRANDS))


@pytest.fixture(scope="module")
def synthetic_panel():
    """Store-week panel generated from known elasticities, store and week effects, and noise."""
    rng = np.random.default_rng(0)
    stores, weeks = 8, 120
    grid = pd.MultiIndex.from_product([range(stores), range(weeks)], names=["store", "week"]).to_frame(index=False)
    log_prices = {b: np.log(0.8) + rng.normal(0, 0.12, len(grid)) for b in BRANDS}
    store_fx, week_fx = rng.normal(0, 0.5, stores), rng.normal(0, 0.3, weeks)
    rows = []
    for brand in BRANDS:
        log_units = (4 + store_fx[grid["store"]] + week_fx[grid["week"]] + rng.normal(0, 0.1, len(grid))
                     + sum(TRUE.loc[brand, other] * log_prices[other] for other in BRANDS))
        frame = grid.assign(brand=brand, units=np.exp(log_units), promo_share=0.0)
        for other in BRANDS:
            frame[f"price_{_slug(other)}"] = np.exp(log_prices[other])
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_recovers_known_elasticities(synthetic_panel):
    fits = dm.fit_all(synthetic_panel, control_promo=False)
    estimated = dm.elasticity_matrix(fits)
    assert np.allclose(estimated.to_numpy(), TRUE.to_numpy(), atol=0.08)


def test_holdout_fit_is_reported(synthetic_panel):
    fit = dm.fit_brand(synthetic_panel, "Dominick's", control_promo=False)
    assert 0 < fit.holdout_r_squared <= 1
    assert fit.n_obs == 8 * 120
