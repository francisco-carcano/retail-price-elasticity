"""Builds a store x week panel of the four chunk light tuna brands from the raw Dominick's files.

Segment: chunk light canned tuna, 6 to 6.5 oz single cans (water and oil). For each brand,
store and week it keeps units sold, average price paid, average unit cost (from the gross
margin field) and the share of units sold on promotion.

Usage: python src/build_panel.py   (writes data/processed/panel.csv)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PANEL_PATH = ROOT / "data" / "processed" / "panel.csv"
BRANDS = ("StarKist", "Chicken of the Sea", "Bumble Bee", "Dominick's")
_BRAND_PREFIXES = (
    (r"STAR|SK ", "StarKist"),
    (r"C O S|COS|CHICKEN", "Chicken of the Sea"),
    (r"BUM", "Bumble Bee"),
    (r"HH|DOM", "Dominick's"),  # Dominick's own label
)
_LIGHT = re.compile(r"CH|CK|CNK|LT|LITE|LIGHT|LGHT")
_NOT_CHUNK_LIGHT = re.compile(r"WHITE|WHT|SOLID|PRIME|SALM|LUNCH|PRE-MIX|MINCED|L/S|LOW S|LS/|TONGOL|HICKORY|CLAM")


def classify_brand(description: str) -> str | None:
    text = description.upper().lstrip("~$ ")
    for pattern, brand in _BRAND_PREFIXES:
        if re.match(pattern, text):
            return brand
    return None


def segment_products(catalogue: pd.DataFrame) -> pd.DataFrame:
    """UPCs in the segment with their brand."""
    description = catalogue["DESCRIP"].str.upper()
    ounces = catalogue["SIZE"].str.extract(r"^(\d+\.?\d*)")[0].astype(float)
    in_segment = (description.str.contains(_LIGHT) & ~description.str.contains(_NOT_CHUNK_LIGHT)
                  & ounces.between(6, 6.5) & ~catalogue["SIZE"].str.contains("/"))
    products = catalogue.loc[in_segment, ["UPC", "DESCRIP"]].copy()
    products["brand"] = products["DESCRIP"].map(classify_brand)
    return products.dropna(subset=["brand"])[["UPC", "brand"]]


def brand_week_sales(movement: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per brand, store and week."""
    m = movement[(movement["OK"] == 1) & (movement["PRICE"] > 0) & (movement["MOVE"] > 0)]
    m = m.merge(products, on="UPC")
    unit_price = m["PRICE"] / m["QTY"]
    m = m.assign(revenue=unit_price * m["MOVE"],
                 cost=unit_price * (1 - m["PROFIT"] / 100) * m["MOVE"],
                 promo_units=m["MOVE"].where(m["SALE"].notna(), 0))
    g = m.groupby(["brand", "STORE", "WEEK"], as_index=False)[["MOVE", "revenue", "cost", "promo_units"]].sum()
    return pd.DataFrame({
        "brand": g["brand"], "store": g["STORE"], "week": g["WEEK"], "units": g["MOVE"],
        "price": g["revenue"] / g["MOVE"], "unit_cost": g["cost"] / g["MOVE"],
        "promo_share": g["promo_units"] / g["MOVE"],
    })


def complete_store_weeks(sales: pd.DataFrame) -> pd.DataFrame:
    """Keeps store-weeks where all four brands sold, so every row has every competitor's price."""
    brands_sold = sales.groupby(["store", "week"])["brand"].transform("nunique")
    return sales[brands_sold == len(BRANDS)].reset_index(drop=True)


def add_competitor_prices(sales: pd.DataFrame) -> pd.DataFrame:
    """Adds one column per brand with that brand's price in the same store-week."""
    prices = sales.pivot(index=["store", "week"], columns="brand", values="price")
    prices.columns = [f"price_{_slug(b)}" for b in prices.columns]
    return sales.merge(prices.reset_index(), on=["store", "week"])


def add_store_demographics(sales: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    """Adds the store's price zone and the log median income of its trading area."""
    d = demographics[["store", "zone", "income"]].dropna(subset=["store"]).astype({"store": int})
    return sales.merge(d.rename(columns={"income": "log_income"}), on="store", how="left")


def _slug(brand: str) -> str:
    return re.sub(r"[^a-z]+", "_", brand.lower()).strip("_")


def build() -> pd.DataFrame:
    catalogue = pd.read_csv(RAW_DIR / "upctna.csv", encoding="latin-1")
    movement = pd.read_csv(RAW_DIR / "wtna" / "wtna.csv",
                           usecols=["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "PROFIT", "OK"])
    demographics = pd.read_stata(RAW_DIR / "demo_stata" / "demo.dta")
    sales = complete_store_weeks(brand_week_sales(movement, segment_products(catalogue)))
    return add_store_demographics(add_competitor_prices(sales), demographics)


if __name__ == "__main__":
    panel = build()
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_PATH, index=False)
    print(f"{len(panel):,} rows, {panel['store'].nunique()} stores, {panel['week'].nunique()} weeks -> {PANEL_PATH}")
