import numpy as np

from export_demo_data import demo_data

RESULTS = {
    "brands": ["a", "b"],
    "elasticities": {"a": {"a": -2.0, "b": 0.4}, "b": {"a": 0.3, "b": -1.5}},
    "covariance": {i: {j: {k: (0.01 if j == k else 0.0) for k in "ab"} for j in "ab"} for i in "ab"},
    "baseline": {b: {"price": 0.8, "units": 100.0, "unit_cost": 0.55, "price_bounds": [0.6, 1.0]} for b in "ab"},
}


def test_demo_data_holds_only_model_outputs():
    data = demo_data(RESULTS | {"brands": ["a", "b"]}, draws=10)
    assert set(data) == {"brands", "private_label", "elasticities", "base_price", "base_units",
                         "unit_cost", "lower", "upper", "draws"}
    assert np.array(data["draws"]).shape == (10, 2, 2)
    assert data["elasticities"] == [[-2.0, 0.4], [0.3, -1.5]]
