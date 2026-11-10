import numpy as np
import pandas as pd

from segment_by_income import split_by_income


def test_split_by_income_halves_stores_and_drops_missing_demographics():
    panel = pd.DataFrame({
        "store": [1, 1, 2, 3, 4, 5],
        "log_income": [10.0, 10.0, 10.5, 11.0, 11.5, np.nan],
    })
    groups = split_by_income(panel)
    low, high = groups["Lower-income stores"], groups["Higher-income stores"]
    assert set(low["store"]) == {1, 2} and set(high["store"]) == {3, 4}
    assert len(low) == 3  # all rows of a store stay together
