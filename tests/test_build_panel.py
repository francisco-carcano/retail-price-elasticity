import math

import pandas as pd
import pytest

import build_panel as bp


@pytest.mark.parametrize("description, brand", [
    ("STAR KIST CH LT TUNA", "StarKist"),
    ("~STAR K CNK LITE TUN", "StarKist"),
    ("C O S CHUNK LIGHT TU", "Chicken of the Sea"),
    ("BUM BEE/WATER CHK LI", "Bumble Bee"),
    ("HH CHUNK LITE TUNA/O", "Dominick's"),
    ("DOM CHUNK LIGHT TUNA", "Dominick's"),
    ("GEISHA CHUNK LT TUNA", None),
])
def test_classify_brand(description, brand):
    assert bp.classify_brand(description) == brand


def test_segment_keeps_only_single_6oz_chunk_light_cans():
    catalogue = pd.DataFrame({
        "UPC": [1, 2, 3, 4, 5],
        "DESCRIP": ["STAR KIST CH LT TUNA", "STAR K SOLID WHT TUN", "STAR KIST CHUNK TUNA",
                    "SK CHNK LGHT TUNA/WA", "BUM BEE OIL CHK LITE"],
        "SIZE": ["6 OZ", "6.12 O", "12.5 O", "3/3 OZ", "6.12 O"],
    })
    assert bp.segment_products(catalogue)["UPC"].tolist() == [1, 5]


def test_brand_week_sales_weights_price_and_cost_by_units():
    movement = pd.DataFrame({
        "STORE": [1, 1, 1], "UPC": [10, 11, 10], "WEEK": [5, 5, 6],
        "MOVE": [30, 10, 0], "QTY": [1, 2, 1], "PRICE": [0.80, 2.00, 0.80],
        "SALE": ["B", None, None], "PROFIT": [25.0, 50.0, 25.0], "OK": [1, 1, 1],
    })
    products = pd.DataFrame({"UPC": [10, 11], "brand": ["StarKist", "StarKist"]})
    row = bp.brand_week_sales(movement, products).iloc[0]  # week 6 has no units and is dropped
    assert row["units"] == 40
    assert math.isclose(row["price"], (30 * 0.80 + 10 * 1.00) / 40)
    assert math.isclose(row["unit_cost"], (30 * 0.60 + 10 * 0.50) / 40)
    assert math.isclose(row["promo_share"], 0.75)


def test_panel_keeps_complete_store_weeks_with_every_competitor_price():
    rows = [("StarKist", 1, 1), ("Chicken of the Sea", 1, 1), ("Bumble Bee", 1, 1), ("Dominick's", 1, 1),
            ("StarKist", 1, 2)]
    sales = pd.DataFrame(rows, columns=["brand", "store", "week"]).assign(
        units=10, price=[0.8, 0.7, 0.9, 0.6, 0.8], unit_cost=0.5, promo_share=0.0)
    panel = bp.add_competitor_prices(bp.complete_store_weeks(sales))
    assert len(panel) == 4 and set(panel["week"]) == {1}
    assert (panel["price_dominick_s"] == 0.6).all()
