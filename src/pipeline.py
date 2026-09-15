"""
Project FORESIGHT — Data Pipeline
==================================
Loads, validates, cleans and joins the four raw data sources into a
single analysis-ready modeling dataset. Also produces a data-quality
report documenting schema/consistency findings.

Run:
    python src/pipeline.py

Outputs:
    data/processed/sales_clean.csv
    data/processed/sku_master_clean.csv
    data/processed/calendar_clean.csv
    data/processed/inventory_clean.csv
    data/processed/modeling_data.csv
    reports/data_quality_report.txt
    reports/data_quality_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"

SALES_FP = DATA_DIR / "sales_daily.csv"
SKU_FP = DATA_DIR / "sku_master.csv"
CAL_FP = DATA_DIR / "calendar.csv"
INV_FP = DATA_DIR / "inventory_snapshots.csv"

# ---------------------------------------------------------------------------
# Expected schemas (column -> dtype hint) — used for validation
# ---------------------------------------------------------------------------
SALES_COLS = ["Date", "SKU", "Units_Sold", "Revenue", "Price", "Promotion"]
SKU_COLS = [
    "SKU", "Product_Name", "Category", "Subcategory",
    "Launch_Date", "Cost_Price", "Selling_Price", "Gross_Margin_Per_Unit",
]
CAL_COLS = [
    "date", "year", "month", "quarter", "week", "day_of_week",
    "is_weekend", "season", "holiday", "is_holiday", "promotion_event",
]
INV_COLS = [
    "Snapshot_Date", "SKU", "Current_Stock", "On_Order", "Lead_Time_Days",
    "Safety_Stock", "Reorder_Point", "Inventory_Value",
]


class DataQualityReport:
    """Collects findings during the pipeline run for later export."""

    def __init__(self) -> None:
        self.sections: dict[str, dict] = {}

    def add(self, section: str, key: str, value) -> None:
        self.sections.setdefault(section, {})[key] = value

    def to_text(self) -> str:
        lines = ["FORESIGHT — DATA QUALITY REPORT", "=" * 40, ""]
        for section, items in self.sections.items():
            lines.append(f"[{section}]")
            for k, v in items.items():
                lines.append(f"  - {k}: {v}")
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.sections, indent=2, default=str)


REPORT = DataQualityReport()


def _fail(msg: str) -> None:
    """Raise a clear pipeline error."""
    raise ValueError(f"[FORESIGHT PIPELINE ERROR] {msg}")


def _validate_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = set(expected) - set(df.columns)
    if missing:
        _fail(f"{name}: missing required columns {sorted(missing)}")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_raw() -> dict[str, pd.DataFrame]:
    for fp in [SALES_FP, SKU_FP, CAL_FP, INV_FP]:
        if not fp.exists():
            _fail(f"Required raw file not found: {fp}")

    sales = pd.read_csv(SALES_FP)
    sku = pd.read_csv(SKU_FP)
    cal = pd.read_csv(CAL_FP)
    inv = pd.read_csv(INV_FP)

    _validate_columns(sales, SALES_COLS, "sales_daily.csv")
    _validate_columns(sku, SKU_COLS, "sku_master.csv")
    _validate_columns(cal, CAL_COLS, "calendar.csv")
    _validate_columns(inv, INV_COLS, "inventory_snapshots.csv")

    return {"sales": sales, "sku": sku, "cal": cal, "inv": inv}


# ---------------------------------------------------------------------------
# 2. Clean individual tables
# ---------------------------------------------------------------------------
def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["SKU"] = df["SKU"].astype(str).str.strip().str.upper()

    for col in ["Units_Sold", "Revenue", "Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Promotion"] = pd.to_numeric(df["Promotion"], errors="coerce").fillna(0).astype(int)

    n_bad_dates = df["Date"].isna().sum()
    n_negative = int(((df["Units_Sold"] < 0) | (df["Revenue"] < 0) | (df["Price"] < 0)).sum())
    n_dupe = int(df.duplicated(subset=["Date", "SKU"]).sum())

    REPORT.add("sales_daily", "rows", len(df))
    REPORT.add("sales_daily", "unique_sku", df["SKU"].nunique())
    REPORT.add("sales_daily", "date_range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    REPORT.add("sales_daily", "unparseable_dates", int(n_bad_dates))
    REPORT.add("sales_daily", "negative_numeric_values", n_negative)
    REPORT.add("sales_daily", "duplicate_date_sku_rows", n_dupe)
    REPORT.add("sales_daily", "nulls_after_cleaning", int(df.isna().sum().sum()))

    return df


def clean_sku_master(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SKU"] = df["SKU"].astype(str).str.strip().str.upper()
    for col in ["Product_Name", "Category", "Subcategory"]:
        df[col] = df[col].astype(str).str.strip()
    df["Launch_Date"] = pd.to_datetime(df["Launch_Date"], errors="coerce")
    for col in ["Cost_Price", "Selling_Price", "Gross_Margin_Per_Unit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_dupe_sku = int(df["SKU"].duplicated().sum())
    n_negative_margin = int((df["Gross_Margin_Per_Unit"] < 0).sum())

    REPORT.add("sku_master", "rows", len(df))
    REPORT.add("sku_master", "unique_sku", df["SKU"].nunique())
    REPORT.add("sku_master", "duplicate_sku_rows", n_dupe_sku)
    REPORT.add(
        "sku_master",
        "skus_with_negative_gross_margin",
        n_negative_margin,
    )
    return df


def clean_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0).astype(int)
    df["is_holiday"] = pd.to_numeric(df["is_holiday"], errors="coerce").fillna(0).astype(int)
    for col in ["season", "day_of_week"]:
        df[col] = df[col].astype(str).str.strip()

    # Expected blanks: no holiday / no promotion event on that date -> "None"
    df["holiday"] = df["holiday"].fillna("None").astype(str).str.strip()
    df["promotion_event"] = df["promotion_event"].fillna("None").astype(str).str.strip()

    n_dupe_dates = int(df["date"].duplicated().sum())
    expected_days = (df["date"].max() - df["date"].min()).days + 1
    n_missing_days = int(expected_days - df["date"].nunique())

    REPORT.add("calendar", "rows", len(df))
    REPORT.add("calendar", "date_range", f"{df['date'].min().date()} to {df['date'].max().date()}")
    REPORT.add("calendar", "duplicate_dates", n_dupe_dates)
    REPORT.add("calendar", "missing_calendar_days_in_range", n_missing_days)
    return df


def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Snapshot_Date"] = pd.to_datetime(df["Snapshot_Date"], errors="coerce")
    df["SKU"] = df["SKU"].astype(str).str.strip().str.upper()
    for col in ["Current_Stock", "On_Order", "Lead_Time_Days", "Safety_Stock",
                "Reorder_Point", "Inventory_Value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_negative = int((df[["Current_Stock", "On_Order", "Safety_Stock",
                          "Reorder_Point", "Inventory_Value"]] < 0).sum().sum())
    n_dupe = int(df.duplicated(subset=["Snapshot_Date", "SKU"]).sum())

    REPORT.add("inventory_snapshots", "rows", len(df))
    REPORT.add("inventory_snapshots", "unique_sku", df["SKU"].nunique())
    REPORT.add(
        "inventory_snapshots",
        "snapshot_date_range",
        f"{df['Snapshot_Date'].min().date()} to {df['Snapshot_Date'].max().date()}",
    )
    REPORT.add("inventory_snapshots", "duplicate_snapshot_sku_rows", n_dupe)
    REPORT.add("inventory_snapshots", "negative_numeric_values", n_negative)
    return df


# ---------------------------------------------------------------------------
# 3. Cross-table relationship checks (the key documented data-quality issue)
# ---------------------------------------------------------------------------
def check_sku_relationships(sales: pd.DataFrame, sku: pd.DataFrame, inv: pd.DataFrame) -> dict:
    sales_skus = set(sales["SKU"].unique())
    master_skus = set(sku["SKU"].unique())
    inv_skus = set(inv["SKU"].unique())

    sales_not_in_master = sorted(sales_skus - master_skus)
    inv_not_in_master = sorted(inv_skus - master_skus)
    valid_skus = sorted(sales_skus & master_skus)

    findings = {
        "sales_sku_count": len(sales_skus),
        "sku_master_count": len(master_skus),
        "inventory_sku_count": len(inv_skus),
        "sales_skus_missing_from_master": sales_not_in_master,
        "inventory_skus_missing_from_master": inv_not_in_master,
        "inventory_skus_missing_from_master_count": len(inv_not_in_master),
        "valid_skus_for_modeling_count": len(valid_skus),
        "valid_skus_for_modeling": valid_skus,
    }

    REPORT.add("sku_relationship_check", "sales_sku_count", findings["sales_sku_count"])
    REPORT.add("sku_relationship_check", "sku_master_count", findings["sku_master_count"])
    REPORT.add("sku_relationship_check", "inventory_sku_count", findings["inventory_sku_count"])
    REPORT.add(
        "sku_relationship_check",
        "inventory_skus_not_in_master_count",
        findings["inventory_skus_missing_from_master_count"],
    )
    REPORT.add(
        "sku_relationship_check",
        "finding",
        "Inventory tracks 200 SKUs but sales/sku_master only cover SKU001-SKU050. "
        "The extra 150 inventory-only SKUs are retained in the cleaned inventory file "
        "but EXCLUDED from the modeling dataset (no sales/master data available for them).",
    )
    REPORT.add(
        "sku_relationship_check", "valid_skus_for_modeling_count", findings["valid_skus_for_modeling_count"]
    )
    return findings


# ---------------------------------------------------------------------------
# 4. Build modeling dataset
# ---------------------------------------------------------------------------
def build_modeling_dataset(
    sales: pd.DataFrame,
    sku: pd.DataFrame,
    cal: pd.DataFrame,
    inv: pd.DataFrame,
    valid_skus: list[str],
) -> pd.DataFrame:
    # Restrict to the 50 SKUs that have both sales and master data.
    sales_valid = sales[sales["SKU"].isin(valid_skus)].copy()
    sku_valid = sku[sku["SKU"].isin(valid_skus)].copy()

    # Safety check: sales rows should be unique per (Date, SKU) and
    # sku_master unique per SKU, and calendar unique per date — this
    # guarantees the following merges are many-to-one, never many-to-many.
    if sales_valid.duplicated(subset=["Date", "SKU"]).any():
        _fail("sales_daily has duplicate Date+SKU rows; cannot safely build 1:1 join.")
    if sku_valid["SKU"].duplicated().any():
        _fail("sku_master has duplicate SKU rows; cannot safely join.")
    if cal["date"].duplicated().any():
        _fail("calendar has duplicate date rows; cannot safely join.")

    df = sales_valid.merge(sku_valid, on="SKU", how="left", validate="many_to_one")
    df = df.merge(cal, left_on="Date", right_on="date", how="left", validate="many_to_one")
    df = df.drop(columns=["date"])

    # --- Attach monthly inventory snapshot info without many-to-many risk ---
    # Inventory is monthly per SKU. Tag both sides with a Year-Month key and
    # verify inventory has exactly one row per (SKU, Year-Month) before joining.
    inv_valid = inv[inv["SKU"].isin(valid_skus)].copy()
    inv_valid["snapshot_month"] = inv_valid["Snapshot_Date"].dt.to_period("M")

    if inv_valid.duplicated(subset=["SKU", "snapshot_month"]).any():
        _fail("inventory_snapshots has >1 row per SKU per month; cannot safely join to daily sales.")

    df["snapshot_month"] = df["Date"].dt.to_period("M")

    inv_cols = ["SKU", "snapshot_month", "Current_Stock", "On_Order",
                "Lead_Time_Days", "Safety_Stock", "Reorder_Point", "Inventory_Value"]
    df = df.merge(
        inv_valid[inv_cols],
        on=["SKU", "snapshot_month"],
        how="left",
        validate="many_to_one",
    )

    n_missing_inv = int(df["Current_Stock"].isna().sum())
    REPORT.add(
        "modeling_dataset",
        "rows_missing_inventory_snapshot_for_month",
        n_missing_inv,
    )

    df = df.drop(columns=["snapshot_month"])
    df = df.sort_values(["SKU", "Date"]).reset_index(drop=True)

    REPORT.add("modeling_dataset", "rows", len(df))
    REPORT.add("modeling_dataset", "columns", len(df.columns))
    REPORT.add("modeling_dataset", "sku_count", df["SKU"].nunique())
    REPORT.add(
        "modeling_dataset",
        "date_range",
        f"{df['Date'].min().date()} to {df['Date'].max().date()}",
    )
    return df


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
def save_outputs(cleaned: dict[str, pd.DataFrame], modeling_df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cleaned["sales"].to_csv(PROCESSED_DIR / "sales_clean.csv", index=False)
    cleaned["sku"].to_csv(PROCESSED_DIR / "sku_master_clean.csv", index=False)
    cleaned["cal"].to_csv(PROCESSED_DIR / "calendar_clean.csv", index=False)
    cleaned["inv"].to_csv(PROCESSED_DIR / "inventory_clean.csv", index=False)
    modeling_df.to_csv(PROCESSED_DIR / "modeling_data.csv", index=False)

    (REPORTS_DIR / "data_quality_report.txt").write_text(REPORT.to_text())
    (REPORTS_DIR / "data_quality_report.json").write_text(REPORT.to_json())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    print("FORESIGHT pipeline starting...")

    raw = load_raw()
    sales = clean_sales(raw["sales"])
    sku = clean_sku_master(raw["sku"])
    cal = clean_calendar(raw["cal"])
    inv = clean_inventory(raw["inv"])

    relationships = check_sku_relationships(sales, sku, inv)
    valid_skus = relationships["valid_skus_for_modeling"]

    modeling_df = build_modeling_dataset(sales, sku, cal, inv, valid_skus)

    save_outputs({"sales": sales, "sku": sku, "cal": cal, "inv": inv}, modeling_df)

    print(f"Done. {len(valid_skus)} SKUs modeled; "
          f"{relationships['inventory_skus_missing_from_master_count']} inventory-only SKUs "
          f"documented but excluded from modeling.")
    print(f"Outputs written to: {PROCESSED_DIR} and {REPORTS_DIR}")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
