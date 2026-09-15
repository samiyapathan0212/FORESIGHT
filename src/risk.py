"""
Project FORESIGHT — Inventory Risk & Decisioning Layer
========================================================
Converts the production demand forecast (selected seasonal-naive baseline,
see reports/forecast_summary.txt) plus the latest inventory position into a
transparent stockout / overstock risk score and recommended action per SKU.

This is business logic, not a machine-learning model: every threshold below
is a documented, configurable assumption.

Run standalone:
    python src/risk.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

FORECAST_FP = PROCESSED_DIR / "forecast_output.csv"
INVENTORY_CLEAN_FP = PROCESSED_DIR / "inventory_clean.csv"
SKU_MASTER_FP = PROCESSED_DIR / "sku_master_clean.csv"

# ---------------------------------------------------------------------------
# Configurable business-rule thresholds (assumptions, not ML outputs)
# ---------------------------------------------------------------------------
OVERSTOCK_MULTIPLIER = 2.0    # flag overstock if Current_Stock > forward_demand_6w * this
FORECAST_HORIZON_WEEKS = 6    # matches the production forecast horizon (notebooks 02/03)


def _fail(msg: str) -> None:
    raise ValueError(f"[FORESIGHT RISK ERROR] {msg}")


# ---------------------------------------------------------------------------
# 1. Load inputs
# ---------------------------------------------------------------------------
def load_forecast() -> pd.DataFrame:
    """Load the production forecast. `predicted_demand` already reflects the
    selected method (seasonal-naive baseline — the HistGradientBoosting model
    did not beat it on backtest, see reports/forecast_summary.txt, and is
    never used here)."""
    if not FORECAST_FP.exists():
        _fail(f"Forecast file not found: {FORECAST_FP}")
    fc = pd.read_csv(FORECAST_FP, parse_dates=["forecast_week"])
    fc = fc.sort_values(["SKU", "forecast_week"]).reset_index(drop=True)
    return fc


def load_latest_inventory() -> pd.DataFrame:
    """Derive exactly one inventory record per SKU from the latest available
    monthly snapshot in inventory_clean.csv (never a future snapshot)."""
    if not INVENTORY_CLEAN_FP.exists():
        _fail(f"Inventory file not found: {INVENTORY_CLEAN_FP}")
    inv = pd.read_csv(INVENTORY_CLEAN_FP, parse_dates=["Snapshot_Date"])
    latest_date = inv["Snapshot_Date"].max()
    latest = inv[inv["Snapshot_Date"] == latest_date].copy()

    if latest["SKU"].duplicated().any():
        _fail("More than one inventory row per SKU in the latest snapshot.")

    latest = latest.drop(columns=["Snapshot_Date"])
    return latest, latest_date


def load_sku_master() -> pd.DataFrame:
    if not SKU_MASTER_FP.exists():
        _fail(f"SKU master file not found: {SKU_MASTER_FP}")
    sku = pd.read_csv(SKU_MASTER_FP)
    if sku["SKU"].duplicated().any():
        _fail("Duplicate SKU rows in sku_master_clean.csv.")
    return sku


# ---------------------------------------------------------------------------
# 2. Build the per-SKU forecast summary (lead-time demand, forward demand)
# ---------------------------------------------------------------------------
def summarize_forecast(fc: pd.DataFrame, lead_time_by_sku: pd.Series) -> pd.DataFrame:
    """For each SKU, compute lead_time_demand (summed over the lead-time
    horizon, capped at the available forecast horizon) and forward_demand_6w
    (summed over the full forecast horizon)."""
    rows = []
    for sku, grp in fc.groupby("SKU"):
        grp = grp.sort_values("forecast_week").reset_index(drop=True)
        n_weeks_available = len(grp)

        lead_time_days = lead_time_by_sku.get(sku, np.nan)
        if pd.isna(lead_time_days):
            lead_time_weeks = n_weeks_available  # safe fallback: no lead-time data
        else:
            lead_time_weeks = math.ceil(lead_time_days / 7)
        lead_time_weeks_capped = min(lead_time_weeks, n_weeks_available)

        lead_time_demand = grp.loc[: lead_time_weeks_capped - 1, "predicted_demand"].sum()
        forward_demand_6w = grp["predicted_demand"].sum()

        rows.append({
            "SKU": sku,
            "lead_time_weeks_used": lead_time_weeks_capped,
            "lead_time_demand": lead_time_demand,
            "forward_demand_6w": forward_demand_6w,
            "avg_weekly_forecast": forward_demand_6w / n_weeks_available if n_weeks_available else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Risk scoring
# ---------------------------------------------------------------------------
def score_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Stockout risk ---
    df["available_supply"] = df["Current_Stock"] + df["On_Order"]
    df["projected_stock_after_lead"] = df["available_supply"] - df["lead_time_demand"]
    df["stockout_risk"] = df["projected_stock_after_lead"] < df["Safety_Stock"]
    df["estimated_units_short"] = (df["Safety_Stock"] - df["projected_stock_after_lead"]).clip(lower=0)

    # --- Overstock risk ---
    zero_demand = df["forward_demand_6w"] <= 0
    df["overstock_risk"] = np.where(
        zero_demand,
        df["Current_Stock"] > 0,
        df["Current_Stock"] > df["forward_demand_6w"] * OVERSTOCK_MULTIPLIER,
    )
    df["overstock_risk"] = df["overstock_risk"].astype(bool)

    # excess_inventory is defined relative to the same OVERSTOCK_MULTIPLIER threshold used
    # for the flag above, so capital_locked_rs only counts stock beyond that threshold
    # (previously this compared against 1x forward demand, inconsistent with the 2x flag).
    df["excess_inventory"] = np.where(
        zero_demand,
        np.where(df["Current_Stock"] > 0, df["Current_Stock"], 0),
        (df["Current_Stock"] - df["forward_demand_6w"] * OVERSTOCK_MULTIPLIER).clip(lower=0),
    )

    # Coverage in weeks, undefined (NaN) when there is no forecast demand to divide by
    df["inventory_coverage_weeks"] = np.where(
        df["avg_weekly_forecast"] > 0,
        df["Current_Stock"] / df["avg_weekly_forecast"],
        np.nan,
    )

    # --- Action mapping (exact truth table from the brief) ---
    action_map = {
        (False, False): "HEALTHY",
        (True, False): "REORDER NOW",
        (False, True): "MARKDOWN / CLEAR",
        (True, True): "WATCH / VOLATILE",
    }
    df["risk_action"] = [action_map[(so, ov)] for so, ov in zip(df["stockout_risk"], df["overstock_risk"])]

    # --- Rupee impact (estimated exposure, not exact financial loss) ---
    df["sales_at_risk_rs"] = df["estimated_units_short"] * df["Selling_Price"]
    df["capital_locked_rs"] = df["excess_inventory"] * df["Cost_Price"]
    df["total_rupee_impact_rs"] = df["sales_at_risk_rs"] + df["capital_locked_rs"]

    # --- Priority score (transparent, additive, each term normalized 0-1ish) ---
    # stockout_severity: units short as a multiple of the safety-stock buffer
    df["stockout_severity"] = df["estimated_units_short"] / (df["Safety_Stock"] + 1)
    # overstock_severity: excess units as a multiple of forward demand
    df["overstock_severity"] = df["excess_inventory"] / (df["forward_demand_6w"] + 1)
    # rupee_severity: this SKU's rupee impact relative to the largest impact in the table
    max_impact = df["total_rupee_impact_rs"].max()
    df["rupee_severity"] = df["total_rupee_impact_rs"] / max_impact if max_impact > 0 else 0.0
    df["priority_score"] = df["stockout_severity"] + df["overstock_severity"] + df["rupee_severity"]

    return df


# ---------------------------------------------------------------------------
# 4. Orchestration
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    "SKU", "Product_Name", "Category", "Current_Stock", "On_Order", "Lead_Time_Days",
    "Safety_Stock", "Reorder_Point", "forward_demand_6w", "lead_time_demand",
    "projected_stock_after_lead", "estimated_units_short", "excess_inventory",
    "inventory_coverage_weeks", "stockout_risk", "overstock_risk", "risk_action",
    "sales_at_risk_rs", "capital_locked_rs", "total_rupee_impact_rs",
    "stockout_severity", "overstock_severity", "rupee_severity", "priority_score",
]


def build_risk_table() -> tuple[pd.DataFrame, pd.Timestamp]:
    fc = load_forecast()
    inv, latest_date = load_latest_inventory()
    sku = load_sku_master()

    modeled_skus = set(fc["SKU"].unique())
    if set(sku["SKU"]) - modeled_skus:
        pass  # sku_master may include SKUs without a forecast; harmless, filtered below by inner merges

    lead_time_by_sku = inv.set_index("SKU")["Lead_Time_Days"]
    forecast_summary = summarize_forecast(fc, lead_time_by_sku)

    # Merge: forecast summary (50 modeled SKUs) + inventory (must have exactly one row per SKU)
    # + sku_master, all on SKU. Use inner joins scoped to modeled SKUs to avoid accidental
    # inclusion of the 150 inventory-only SKUs documented in the pipeline's data-quality report.
    table = forecast_summary.merge(inv, on="SKU", how="left", validate="one_to_one")
    table = table.merge(
        sku[["SKU", "Product_Name", "Category", "Cost_Price", "Selling_Price"]],
        on="SKU", how="left", validate="one_to_one",
    )

    if table["SKU"].duplicated().any():
        _fail("Duplicate SKU rows after merging forecast + inventory + sku_master.")
    if len(table) != len(modeled_skus):
        _fail(f"Expected {len(modeled_skus)} risk rows (one per modeled SKU), got {len(table)}.")
    if table[["Current_Stock", "On_Order", "Safety_Stock", "Cost_Price", "Selling_Price"]].isna().any().any():
        _fail("Missing inventory or pricing values after merge — cannot score risk safely.")

    table = score_risk(table)
    table = table.sort_values("priority_score", ascending=False).reset_index(drop=True)

    return table[OUTPUT_COLUMNS], latest_date


def save_outputs(table: pd.DataFrame, latest_date: pd.Timestamp) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    table.to_csv(PROCESSED_DIR / "risk_output.csv", index=False)

    action_counts = table["risk_action"].value_counts()
    metrics = pd.DataFrame({
        "metric": [
            "latest_inventory_snapshot_date", "overstock_multiplier", "forecast_horizon_weeks",
            "n_skus", "n_reorder_now", "n_markdown_clear", "n_watch_volatile", "n_healthy",
            "total_sales_at_risk_rs", "total_capital_locked_rs", "total_rupee_impact_rs",
        ],
        "value": [
            str(latest_date.date()), OVERSTOCK_MULTIPLIER, FORECAST_HORIZON_WEEKS,
            len(table),
            int(action_counts.get("REORDER NOW", 0)),
            int(action_counts.get("MARKDOWN / CLEAR", 0)),
            int(action_counts.get("WATCH / VOLATILE", 0)),
            int(action_counts.get("HEALTHY", 0)),
            round(table["sales_at_risk_rs"].sum(), 2),
            round(table["capital_locked_rs"].sum(), 2),
            round(table["total_rupee_impact_rs"].sum(), 2),
        ],
    })
    metrics.to_csv(REPORTS_DIR / "risk_metrics.csv", index=False)

    with open(REPORTS_DIR / "risk_summary.txt", "w") as f:
        f.write("FORESIGHT — INVENTORY RISK SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Latest inventory snapshot used: {latest_date.date()}\n")
        f.write(f"Forecast used: production forecast (selected seasonal-naive baseline)\n")
        f.write(f"Overstock multiplier (business rule, configurable): {OVERSTOCK_MULTIPLIER}\n\n")
        f.write("SKU counts by action:\n")
        for action in ["REORDER NOW", "WATCH / VOLATILE", "MARKDOWN / CLEAR", "HEALTHY"]:
            f.write(f"  {action}: {int(action_counts.get(action, 0))}\n")
        f.write(f"\nTotal estimated sales at risk (stockouts): Rs {table['sales_at_risk_rs'].sum():,.0f}\n")
        f.write(f"Total estimated capital locked (overstock): Rs {table['capital_locked_rs'].sum():,.0f}\n")
        f.write(f"Total estimated rupee impact: Rs {table['total_rupee_impact_rs'].sum():,.0f}\n\n")
        f.write("Note: these are estimated exposure figures based on documented business-rule "
                "assumptions (see OVERSTOCK_MULTIPLIER and the lead-time/safety-stock logic in "
                "src/risk.py), not confirmed financial losses.\n")


def run() -> None:
    table, latest_date = build_risk_table()
    save_outputs(table, latest_date)
    print(f"Risk table built for {len(table)} SKUs (inventory snapshot: {latest_date.date()}).")
    print(table["risk_action"].value_counts().to_string())
    print(f"Total rupee impact: Rs {table['total_rupee_impact_rs'].sum():,.0f}")


if __name__ == "__main__":
    run()
