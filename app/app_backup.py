"""
Project FORESIGHT — Demand & Inventory Intelligence
Streamlit dashboard (productization layer).

Consumes the already-computed pipeline / EDA / forecast / risk outputs.
Does NOT recompute forecasting or risk logic — see src/pipeline.py and src/risk.py.

Run:
    streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

ACTION_COLORS = {
    "REORDER NOW": "#EF4444",
    "WATCH / VOLATILE": "#F59E0B",
    "MARKDOWN / CLEAR": "#3B82F6",
    "HEALTHY": "#22C55E",
}
ACTION_ORDER = ["REORDER NOW", "WATCH / VOLATILE", "MARKDOWN / CLEAR", "HEALTHY"]
ACTION_CSS = {"REORDER NOW": "kpi-danger", "WATCH / VOLATILE": "kpi-warn",
              "MARKDOWN / CLEAR": "kpi-accent", "HEALTHY": "kpi-good"}
RECENT_WEEKS_DEFAULT = 18  # default zoom window for time-series charts (full history via Plotly zoom)

st.set_page_config(
    page_title="FORESIGHT — Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme (premium dark)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
    --bg: #0B0F1E;
    --panel: #131A2E;
    --panel-2: #171F36;
    --border: #262F49;
    --text: #E7EAF3;
    --muted: #8B93AD;
    --accent: #7C6CF6;
    --accent-2: #4F7DFA;
}
.stApp { background: linear-gradient(180deg, #0B0F1E 0%, #0D1224 100%); color: var(--text); }
section[data-testid="stSidebar"] { background: #0D1226; border-right: 1px solid var(--border); width: 21rem !important; }
h1, h2, h3, h4, p, span, label, div { color: var(--text); }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1320px; overflow-x: hidden; }

.foresight-header {
    padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #171F3A 0%, #11162A 100%);
    border: 1px solid var(--border);
}
.foresight-title { font-size: 2rem; font-weight: 800; letter-spacing: 0.5px;
    background: linear-gradient(90deg, #A78BFA, #7C6CF6, #4F7DFA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
.foresight-sub { color: var(--muted); font-size: 0.98rem; margin-top: 0.35rem; }

.page-header { margin-bottom: 1.1rem; }
.page-header h2 { font-size: 1.5rem; font-weight: 750; margin: 0; color: var(--text); }
.page-header p { color: var(--muted); font-size: 0.92rem; margin: 0.2rem 0 0 0; }

.kpi-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 1.0rem 1.1rem; box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    min-height: 100px; display: flex; flex-direction: column; justify-content: center;
    overflow: hidden;
}
.kpi-label { color: var(--muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.5px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi-value { font-size: clamp(1.05rem, 1.7vw, 1.55rem); font-weight: 780; margin-top: 0.28rem;
    line-height: 1.15; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi-subtitle { color: var(--muted); font-size: 0.75rem; margin-top: 0.28rem; }
.kpi-accent { color: var(--accent-2); }
.kpi-danger { color: #F87171; }
.kpi-warn { color: #FBBF24; }
.kpi-good { color: #4ADE80; }

.section-title { font-size: 1.1rem; font-weight: 700; margin: 1.2rem 0 0.15rem 0; color: var(--text); }
.section-sub { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.6rem; }
.chart-caption { color: var(--muted); font-size: 0.79rem; margin-top: -0.3rem; margin-bottom: 0.5rem; }
.group-label { font-size: 0.72rem; font-weight: 750; letter-spacing: 0.9px; text-transform: uppercase;
    color: var(--accent-2); margin: 1.5rem 0 0.5rem 0; border-bottom: 1px solid var(--border); padding-bottom: 0.35rem; }

.badge { display: inline-block; padding: 0.22rem 0.65rem; border-radius: 999px;
    font-size: 0.74rem; font-weight: 650; border: 1px solid var(--border); white-space: nowrap; }
.badge-model { background: rgba(124,108,246,0.15); color: #C4B5FD; }
.badge-date { background: rgba(79,125,250,0.15); color: #93C5FD; }

.action-pill { padding: 0.18rem 0.65rem; border-radius: 999px; font-size: 0.76rem; font-weight: 700; white-space: nowrap; }

.insight-box {
    background: var(--panel-2); border-left: 3px solid var(--accent);
    border-radius: 10px; padding: 0.9rem 1.15rem; margin-top: 0.5rem; line-height: 1.55; font-size: 0.93rem;
}
.why-box {
    background: linear-gradient(135deg, rgba(124,108,246,0.14), rgba(79,125,250,0.08));
    border: 1px solid rgba(124,108,246,0.35); border-radius: 12px;
    padding: 1rem 1.3rem; margin-top: 0.7rem; line-height: 1.6; font-size: 0.95rem;
}
.why-label { color: #C4B5FD; font-weight: 750; text-transform: uppercase; font-size: 0.73rem;
    letter-spacing: 0.6px; margin-bottom: 0.3rem; display: block; }

.snapshot-box {
    background: linear-gradient(135deg, #1B2440 0%, #131A2E 100%);
    border: 1px solid var(--border); border-radius: 16px; padding: 1.2rem 1.4rem;
    margin: 0.6rem 0 1.2rem 0; box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.snapshot-title { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.7px;
    color: var(--muted); margin-bottom: 0.5rem; font-weight: 650; }
.snapshot-line { font-size: 1.0rem; line-height: 1.8; }
.snapshot-num { font-weight: 780; }

.mgmt-box {
    background: linear-gradient(135deg, rgba(124,108,246,0.10), rgba(79,125,250,0.05));
    border: 1px solid rgba(124,108,246,0.30); border-radius: 16px; padding: 1.2rem 1.4rem;
    margin: 0.7rem 0 1.2rem 0;
}
.mgmt-title { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.7px; color: #C4B5FD;
    font-weight: 700; margin-bottom: 0.6rem; }
.mgmt-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.6rem 1.4rem; }
.mgmt-item { font-size: 0.96rem; line-height: 1.5; }
.mgmt-num { font-weight: 800; font-size: 1.05rem; }
.mgmt-num.danger { color: #F87171; } .mgmt-num.warn { color: #FBBF24; } .mgmt-num.good { color: #4ADE80; }
.mgmt-num.accent { color: #93C5FD; }

.subsection-label { font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.6px;
    color: var(--accent-2); font-weight: 700; margin-bottom: 0.45rem; }

.rank-badge { display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%; background: rgba(124,108,246,0.2);
    color: #C4B5FD; font-size: 0.72rem; font-weight: 750; margin-right: 0.4rem; }

.flow-grid { display: grid; grid-template-columns: 3fr 0.6fr 3fr 0.6fr 3fr 0.6fr 3fr;
    align-items: stretch; gap: 0.4rem; margin: 0.7rem 0; }
.flow-box { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 0.9rem 0.8rem; text-align: center; font-size: 0.84rem; font-weight: 600;
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.25rem; }
.flow-box .flow-head { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); }
.flow-box.highlight { border-color: rgba(124,108,246,0.5); background: rgba(124,108,246,0.1); color: #C4B5FD; }
.flow-box.outcome { border-color: rgba(34,197,94,0.5); background: rgba(34,197,94,0.08); color: #86EFAC; }
.flow-arrow { display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 1.4rem; }
@media (max-width: 900px) {
    .flow-grid { grid-template-columns: 1fr; }
    .flow-arrow { transform: rotate(90deg); }
}

.mini-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.7rem; margin-top: 0.5rem; }
.mini-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 0.85rem 1rem; font-size: 0.88rem; line-height: 1.45; }
.mini-card .mini-icon { font-size: 1.05rem; margin-right: 0.4rem; }

.status-banner { border-radius: 12px; padding: 0.9rem 1.2rem; margin: 0.8rem 0 1rem 0;
    font-weight: 750; font-size: 1.0rem; display: flex; align-items: center; justify-content: space-between; }
.status-banner.action { background: rgba(239,68,68,0.14); border: 1px solid rgba(239,68,68,0.4); color: #FCA5A5; }
.status-banner.ok { background: rgba(34,197,94,0.14); border: 1px solid rgba(34,197,94,0.4); color: #86EFAC; }

.why-list { margin: 0; padding-left: 1.1rem; line-height: 1.85; }

.table-wrap { width: 100%; overflow-x: auto; border-radius: 10px; }

section[data-testid="stSidebar"] label[data-baseweb="radio"] { padding: 0.12rem 0; }
section[data-testid="stSidebar"] .stRadio label p { font-size: 0.92rem; }

hr { border-color: var(--border); }
[data-testid="stMetricValue"] { color: var(--text); }
table { width: 100%; border-collapse: collapse; }
table th { text-align: left; color: var(--muted); font-size: 0.76rem; text-transform: uppercase;
    letter-spacing: 0.4px; padding: 0.5rem 0.65rem; border-bottom: 1px solid var(--border); white-space: nowrap; }
table td { padding: 0.5rem 0.65rem; border-bottom: 1px solid rgba(38,47,73,0.6); font-size: 0.89rem; white-space: nowrap; }
table tr:hover td { background: rgba(124,108,246,0.06); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached, validated)
# ---------------------------------------------------------------------------
REQUIRED_FILES = {
    "modeling_data.csv": ["Date", "SKU", "Units_Sold", "Revenue", "Category", "Promotion", "season", "is_holiday"],
    "sku_performance_summary.csv": ["SKU", "Product_Name", "Units_Sold", "Revenue",
                                     "Coefficient_of_Variation", "Gross_Margin_Per_Unit"],
    "inventory_latest_snapshot_summary.csv": ["SKU", "Current_Stock", "On_Order", "Safety_Stock",
                                               "Reorder_Point", "Inventory_Value", "Days_of_Stock_On_Hand"],
    "forecast_output.csv": ["SKU", "forecast_week", "predicted_demand", "baseline_prediction", "selected_model"],
    "risk_output.csv": ["SKU", "Product_Name", "Category", "Current_Stock", "On_Order", "Lead_Time_Days",
                         "Safety_Stock", "Reorder_Point", "forward_demand_6w", "lead_time_demand",
                         "projected_stock_after_lead", "inventory_coverage_weeks", "stockout_risk",
                         "overstock_risk", "risk_action", "sales_at_risk_rs", "capital_locked_rs",
                         "total_rupee_impact_rs", "overstock_severity", "stockout_severity", "priority_score"],
}


@st.cache_data(show_spinner=False)
def load_all_data():
    missing_files, missing_cols = [], {}
    frames = {}
    for fname, cols in REQUIRED_FILES.items():
        fp = PROCESSED_DIR / fname
        if not fp.exists():
            missing_files.append(fname)
            continue
        df = pd.read_csv(fp)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            missing_cols[fname] = missing
        frames[fname] = df

    if missing_files or missing_cols:
        return None, missing_files, missing_cols

    frames["modeling_data.csv"]["Date"] = pd.to_datetime(frames["modeling_data.csv"]["Date"])
    frames["forecast_output.csv"]["forecast_week"] = pd.to_datetime(frames["forecast_output.csv"]["forecast_week"])

    risk = frames["risk_output.csv"]
    if risk["SKU"].duplicated().any():
        missing_cols["risk_output.csv"] = ["duplicate SKU rows detected"]
        return None, missing_files, missing_cols

    return frames, [], {}


@st.cache_data(show_spinner=False)
def load_text_report(name: str) -> str:
    fp = REPORTS_DIR / name
    return fp.read_text() if fp.exists() else ""


@st.cache_data(show_spinner=False)
def load_forecast_metrics():
    fp = REPORTS_DIR / "forecast_metrics.csv"
    return pd.read_csv(fp) if fp.exists() else None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_rs(x: float) -> str:
    if pd.isna(x):
        return "—"
    if abs(x) >= 1e7:
        return f"₹{x/1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x/1e5:,.2f} L"
    return f"₹{x:,.0f}"


def fmt_units(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}"


def fmt_days(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.0f} days"


def fmt_date(x) -> str:
    try:
        return pd.to_datetime(x).strftime("%d %b %Y")
    except (ValueError, TypeError):
        return str(x)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = ""):
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="page-header"><h2>{title}</h2>{sub_html}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, css_class: str = "kpi-accent", subtitle: str | None = None):
    sub_html = f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="kpi-card" title="{value}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {css_class}">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def action_pill_html(action: str) -> str:
    color = ACTION_COLORS.get(action, "#8B93AD")
    return f'<span class="action-pill" style="background:{color}22;color:{color};border:1px solid {color}55;">{action}</span>'


def caption(text: str):
    st.markdown(f'<div class="chart-caption">{text}</div>', unsafe_allow_html=True)


def section(title: str, sub: str | None = None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def group_label(text: str):
    st.markdown(f'<div class="group-label">{text}</div>', unsafe_allow_html=True)


def dark_layout(fig, height=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       height=height, margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
    return fig


def render_table(html: str):
    st.markdown(f'<div class="table-wrap">{html}</div>', unsafe_allow_html=True)


def insight_cards(items: list[str], columns: int = 3, icon: str = "▸"):
    """Render a grid of compact insight cards instead of one long bullet block."""
    html = '<div class="mini-card-grid">' + "".join(
        f'<div class="mini-card"><span class="mini-icon">{icon}</span>{item}</div>' for item in items
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def compact_filters(options_df: pd.DataFrame, cat_col="Category", sku_col="SKU", key_prefix="f"):
    """Compact 'All Categories' / 'All SKUs' dropdown filters. Returns (categories, skus)."""
    all_categories = sorted(options_df[cat_col].dropna().unique())
    c1, c2 = st.columns(2)
    with c1:
        cat_choice = st.selectbox("Category", ["All Categories"] + all_categories, key=f"{key_prefix}_cat")
    sel_categories = all_categories if cat_choice == "All Categories" else [cat_choice]
    sku_pool = sorted(options_df.loc[options_df[cat_col].isin(sel_categories), sku_col].unique())
    with c2:
        sku_choice = st.selectbox("SKU", ["All SKUs"] + sku_pool, key=f"{key_prefix}_sku")
    sel_skus = sku_pool if sku_choice == "All SKUs" else [sku_choice]
    return sel_categories, sel_skus


def quadrant_decision_matrix(df: pd.DataFrame, height: int = 440):
    """Stockout-vs-overstock risk matrix with shaded quadrants and labels.
    x = overstock_severity, y = stockout_severity. Severity is 0 exactly when the
    corresponding boolean risk flag is False (see src/risk.py), so the x=0/y=0 lines
    are the true quadrant boundaries used for the risk_action classification."""
    if df.empty:
        return None

    x_max = max(df["overstock_severity"].max() * 1.10, 0.08)
    y_max = max(df["stockout_severity"].max() * 1.10, 0.08)

    fig = px.scatter(
        df, x="overstock_severity", y="stockout_severity", color="risk_action",
        size="total_rupee_impact_rs", hover_name="Product_Name",
        hover_data={"SKU": True, "Category": True, "total_rupee_impact_rs": ":,.0f",
                    "overstock_severity": ":.2f", "stockout_severity": ":.2f"},
        color_discrete_map=ACTION_COLORS, category_orders={"risk_action": ACTION_ORDER},
        labels={"overstock_severity": "Overstock Severity →", "stockout_severity": "Stockout Severity ↑",
                "risk_action": "Action"},
        template="plotly_dark", size_max=30,
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.18)")))

    quadrants = [
        (0, x_max, 0, y_max, "rgba(245,158,11,0.10)", "WATCH / VOLATILE", x_max * 0.97, y_max * 0.95),
        (-x_max * 0.015, 0, 0, y_max, "rgba(239,68,68,0.10)", "REORDER NOW", x_max * 0.015, y_max * 0.95),
        (0, x_max, -y_max * 0.015, 0, "rgba(59,130,246,0.10)", "MARKDOWN / CLEAR", x_max * 0.97, y_max * 0.05),
        (-x_max * 0.015, 0, -y_max * 0.015, 0, "rgba(34,197,94,0.10)", "HEALTHY", x_max * 0.015, y_max * 0.05),
    ]
    for x0, x1, y0, y1, color, label, lx, ly in quadrants:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below")
        fig.add_annotation(x=lx, y=ly, text=f"<b>{label}</b>", showarrow=False,
                            font=dict(size=11, color="rgba(231,234,243,0.55)"),
                            xanchor="right" if lx > x_max / 2 else "left",
                            yanchor="top" if ly > y_max / 2 else "bottom")

    fig.add_vline(x=0, line_width=1, line_color="rgba(231,234,243,0.25)")
    fig.add_hline(y=0, line_width=1, line_color="rgba(231,234,243,0.25)")
    fig.update_xaxes(range=[-x_max * 0.02, x_max])
    fig.update_yaxes(range=[-y_max * 0.02, y_max])
    dark_layout(fig, height=height)
    fig.update_layout(margin=dict(t=25, b=10, l=10, r=10))
    return fig


def priority_table_html(df: pd.DataFrame, with_rank: bool = True) -> str:
    t = df.copy().reset_index(drop=True)
    t["Action"] = t["risk_action"].apply(action_pill_html)
    t["₹ Impact"] = t["total_rupee_impact_rs"].apply(fmt_rs)
    t["Priority Score"] = t["priority_score"].round(2)
    t = t.rename(columns={"Product_Name": "Product"})
    cols = ["SKU", "Product", "Category", "Action", "Priority Score", "₹ Impact"]
    if with_rank:
        t.insert(0, "Rank", [f'<span class="rank-badge">{i+1}</span>' for i in range(len(t))])
        cols = ["Rank"] + cols
    return t[cols].to_html(escape=False, index=False)


def actual_vs_forecast_chart(sku: str, height: int = 380, recent_weeks: int | None = RECENT_WEEKS_DEFAULT):
    m_sku = modeling_df[modeling_df["SKU"] == sku].copy()
    m_sku["week_start"] = m_sku["Date"] - pd.to_timedelta(m_sku["Date"].dt.dayofweek, unit="D")
    weekly_sku = m_sku.groupby("week_start")["Units_Sold"].sum().reset_index()
    fc_sku = forecast_df[forecast_df["SKU"] == sku].sort_values("forecast_week")
    forecast_start = fc_sku["forecast_week"].min() if not fc_sku.empty else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_sku["week_start"], y=weekly_sku["Units_Sold"],
                              mode="lines", name="Historical Actual", line=dict(color="#4F7DFA", width=2.3)))
    if not fc_sku.empty:
        fig.add_trace(go.Scatter(x=fc_sku["forecast_week"], y=fc_sku["baseline_prediction"],
                                  mode="lines+markers", name="Future Forecast (Seasonal-Naive)",
                                  line=dict(color="#FBBF24", width=2.6, dash="dash"),
                                  marker=dict(size=7, symbol="diamond")))
    if forecast_start is not None:
        fig.add_vline(x=forecast_start, line_width=1.6, line_dash="dot", line_color="#FBBF24")
        fig.add_annotation(x=forecast_start, y=1, yref="paper", text="FORECAST STARTS",
                            showarrow=False, font=dict(size=10, color="#FBBF24"), yanchor="bottom")

    if recent_weeks is not None and not weekly_sku.empty:
        default_start = weekly_sku["week_start"].max() - pd.Timedelta(weeks=recent_weeks)
        default_end = fc_sku["forecast_week"].max() if not fc_sku.empty else weekly_sku["week_start"].max()
        fig.update_xaxes(range=[default_start, default_end])

    fig.update_layout(xaxis_title="Week", yaxis_title="Units Sold")
    dark_layout(fig, height=height)
    return fig


def why_flagged_text(row: pd.Series) -> str:
    if row["risk_action"] == "REORDER NOW":
        return ("Forecast demand during the replenishment lead time exceeds available supply "
                "(on-hand + on-order), and projected stock after that lead time falls below the safety-stock level.")
    if row["risk_action"] == "MARKDOWN / CLEAR":
        return ("Current inventory exceeds the configured 2× six-week forecast threshold — "
                "more stock is on hand than is expected to sell well beyond the forecast window.")
    if row["risk_action"] == "WATCH / VOLATILE":
        return "Both stockout and overstock conditions are present — the position should be reviewed manually."
    return "Stock position is within safe and efficient bounds for the forecasted demand — no action needed."


def status_banner(action: str):
    if action == "HEALTHY":
        st.markdown(f'<div class="status-banner ok"><span>✅ NO ACTION REQUIRED</span>{action_pill_html(action)}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-banner action"><span>⚠️ ACTION REQUIRED</span>{action_pill_html(action)}</div>',
                    unsafe_allow_html=True)


def model_comparison_cards(baseline_wape, ml_wape):
    winner = baseline_wape is not None and ml_wape is not None and baseline_wape <= ml_wape
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="min-height:auto; padding:1rem 1.1rem; border-color: rgba(34,197,94,0.4);">
            <div class="kpi-label">Seasonal-Naive</div>
            <div class="kpi-value kpi-good">{f"{baseline_wape*100:.2f}%" if baseline_wape is not None else "n/a"}</div>
            <div class="kpi-subtitle" style="color:#4ADE80; font-weight:700;">✓ SELECTED — PRODUCTION MODEL</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="min-height:auto; padding:1rem 1.1rem; border-color: rgba(239,68,68,0.3); opacity:0.85;">
            <div class="kpi-label">Gradient Boosting</div>
            <div class="kpi-value kpi-danger">{f"{ml_wape*100:.2f}%" if ml_wape is not None else "n/a"}</div>
            <div class="kpi-subtitle" style="color:#F87171; font-weight:700;">✗ NOT SELECTED</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div class="insight-box">The evaluated gradient-boosting model did not outperform the '
                'seasonal-naive baseline on backtest, so Seasonal-Naive is used for production forecasting.</div>',
                unsafe_allow_html=True)


def get_latest_snapshot_date() -> str:
    """Derive the latest inventory snapshot date from the risk summary report
    (falls back to the year/month columns in the inventory summary CSV)."""
    summary_text = load_text_report("risk_summary.txt")
    for line in summary_text.splitlines():
        if "latest inventory snapshot" in line.lower():
            date_str = line.split(":", 1)[-1].strip()
            try:
                return pd.to_datetime(date_str).strftime("%d %b %Y")
            except (ValueError, TypeError):
                pass
    if "year" in inv_latest.columns and "month" in inv_latest.columns and not inv_latest.empty:
        y, mo = int(inv_latest["year"].iloc[0]), int(inv_latest["month"].iloc[0])
        return pd.Timestamp(year=y, month=mo, day=1).strftime("%d %b %Y")
    return "n/a"


# ---------------------------------------------------------------------------
# Load data up front
# ---------------------------------------------------------------------------
data, missing_files, missing_cols = load_all_data()

if data is None:
    st.markdown("""
    <div class="foresight-header">
        <p class="foresight-title">FORESIGHT</p>
        <p class="foresight-sub">Demand &amp; Inventory Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    st.error("The dashboard could not load its required data files.")
    if missing_files:
        st.write("**Missing files:**", missing_files)
    if missing_cols:
        st.write("**Missing/invalid columns:**", missing_cols)
    st.info("Run `python src/pipeline.py`, the notebooks, and `python src/risk.py` first, "
            "then reload this app.")
    st.stop()

modeling_df = data["modeling_data.csv"]
sku_perf = data["sku_performance_summary.csv"]
inv_latest = data["inventory_latest_snapshot_summary.csv"]
forecast_df = data["forecast_output.csv"]
risk_df = data["risk_output.csv"]
forecast_metrics = load_forecast_metrics()

# Enrich the inventory snapshot with Category (from risk_output — same 50 SKUs, one-to-one)
inv_enriched = inv_latest.merge(risk_df[["SKU", "Category"]], on="SKU", how="left", validate="one_to_one")

baseline_wape = ml_wape = None
if forecast_metrics is not None:
    b_row = forecast_metrics[forecast_metrics["Method"].str.contains("Seasonal-Naive", case=False, na=False)]
    m_row = forecast_metrics[~forecast_metrics["Method"].str.contains("Seasonal-Naive", case=False, na=False)]
    if not b_row.empty:
        baseline_wape = b_row.iloc[0]["WAPE"]
    if not m_row.empty:
        ml_wape = m_row.iloc[0]["WAPE"]

latest_snapshot_date = get_latest_snapshot_date()

# ---------------------------------------------------------------------------
# Sidebar navigation (compact system status only)
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 📦 FORESIGHT")
PAGES = ["🏠 Home", "📊 Sales Analytics", "🔮 Forecast", "📦 Inventory Dashboard",
         "⚠️ Risk Dashboard", "🔎 Product Details", "🎯 Executive Summary"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown('<span class="badge badge-model">Production model: Seasonal-Naive</span>', unsafe_allow_html=True)
if baseline_wape is not None:
    st.sidebar.markdown(f'<span class="badge badge-model">WAPE: {baseline_wape*100:.2f}%</span>', unsafe_allow_html=True)
st.sidebar.markdown(f'<span class="badge badge-date">Latest inventory snapshot: {latest_snapshot_date}</span>',
                     unsafe_allow_html=True)


# ===========================================================================
# PAGE — HOME
# ===========================================================================
def page_home():
    st.markdown("""
    <div class="foresight-header">
        <p class="foresight-title">FORESIGHT</p>
        <p class="foresight-sub">Demand &amp; Inventory Intelligence — Forecast demand. Detect inventory risk. Prioritize action.</p>
    </div>
    """, unsafe_allow_html=True)

    total_revenue = modeling_df["Revenue"].sum()
    total_units = modeling_df["Units_Sold"].sum()
    sales_at_risk = risk_df["sales_at_risk_rs"].sum()
    capital_locked = risk_df["capital_locked_rs"].sum()
    n_action = int((risk_df["risk_action"] != "HEALTHY").sum())
    n_reorder = int((risk_df["risk_action"] == "REORDER NOW").sum())
    n_markdown = int((risk_df["risk_action"] == "MARKDOWN / CLEAR").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Total Revenue", fmt_rs(total_revenue))
    with c2: kpi_card("Total Units Sold", fmt_units(total_units))
    with c3: kpi_card("Sales at Risk", fmt_rs(sales_at_risk), "kpi-danger")
    with c4: kpi_card("Capital Locked", fmt_rs(capital_locked), "kpi-warn")
    with c5: kpi_card("SKUs Requiring Action", f"{n_action} / {len(risk_df)}",
                       "kpi-warn" if n_action else "kpi-good")

    st.markdown(f"""
    <div class="mgmt-box">
        <div class="mgmt-title">Management Snapshot</div>
        <div class="mgmt-grid">
            <div class="mgmt-item"><span class="mgmt-num warn">{n_action} of {len(risk_df)}</span> SKUs require action</div>
            <div class="mgmt-item"><span class="mgmt-num danger">{n_reorder}</span> REORDER NOW</div>
            <div class="mgmt-item"><span class="mgmt-num accent">{n_markdown}</span> MARKDOWN / CLEAR</div>
            <div class="mgmt-item"><span class="mgmt-num danger">{fmt_rs(sales_at_risk)}</span> estimated sales exposure</div>
            <div class="mgmt-item"><span class="mgmt-num warn">{fmt_rs(capital_locked)}</span> estimated capital exposure</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section("How FORESIGHT Works")
    st.markdown("""
    <div class="flow-grid">
        <div class="flow-box"><span class="flow-head">Data Sources</span>Sales · Product<br>Calendar · Inventory</div>
        <div class="flow-arrow">→</div>
        <div class="flow-box highlight"><span class="flow-head">Forecast</span>6-week demand forecast<br>Seasonal-Naive</div>
        <div class="flow-arrow">→</div>
        <div class="flow-box highlight"><span class="flow-head">Risk Engine</span>Stockout risk · Overstock risk<br>Priority scoring</div>
        <div class="flow-arrow">→</div>
        <div class="flow-box outcome"><span class="flow-head">Action</span>Reorder Now · Markdown/Clear<br>Healthy</div>
    </div>
    """, unsafe_allow_html=True)

    section("Why It Matters")
    insight_cards([
        "<b>Prevent stockouts</b> on fast-moving products before they run out",
        "<b>Reduce excess inventory</b> that ties up working capital",
        "<b>Protect revenue</b> by acting before demand shifts",
        "<b>Free working capital</b> locked in slow-moving stock",
    ], columns=4, icon="✓")


# ===========================================================================
# PAGE — SALES ANALYTICS
# ===========================================================================
def page_sales_analytics():
    page_header("Sales Analytics", "What has happened historically?")

    total_revenue = modeling_df["Revenue"].sum()
    total_units = modeling_df["Units_Sold"].sum()
    c1, c2 = st.columns(2)
    with c1: kpi_card("Total Revenue", fmt_rs(total_revenue))
    with c2: kpi_card("Total Units Sold", fmt_units(total_units))

    # Pre-compute aggregates used both by charts and insights
    weekly = modeling_df.copy()
    weekly["week_start"] = weekly["Date"] - pd.to_timedelta(weekly["Date"].dt.dayofweek, unit="D")
    weekly_agg = weekly.groupby("week_start").agg(Revenue=("Revenue", "sum"), Units_Sold=("Units_Sold", "sum")).reset_index()
    cat_rev = modeling_df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    season_order = ["Winter", "Spring", "Summer", "Monsoon", "Autumn"]
    season_perf = modeling_df.groupby("season")["Units_Sold"].sum().reindex(
        [s for s in season_order if s in modeling_df["season"].unique()])
    promo_compare = modeling_df.groupby("Promotion")["Units_Sold"].mean().rename(
        index={0: "Non-Promotion", 1: "Promotion"}).reset_index()
    top_sku_row = sku_perf.sort_values("Revenue", ascending=False).iloc[0]
    most_volatile = sku_perf.sort_values("Coefficient_of_Variation", ascending=False).iloc[0]
    promo_avg = promo_compare.set_index("Promotion")["Units_Sold"]
    promo_lift = None
    if "Promotion" in promo_avg.index and "Non-Promotion" in promo_avg.index:
        promo_lift = 100 * (promo_avg["Promotion"] / promo_avg["Non-Promotion"] - 1)

    section("Key Insights")
    insight_items = [
        f"<b>{season_perf.idxmax()}</b> generated the highest total demand historically",
        f"<b>{cat_rev.index[0]}</b> is the strongest category by revenue ({fmt_rs(cat_rev.iloc[0])})",
        f"<b>{top_sku_row['Product_Name']}</b> ({top_sku_row['SKU']}) is the top revenue SKU "
        f"({fmt_rs(top_sku_row['Revenue'])})",
        f"<b>{most_volatile['Product_Name']}</b> ({most_volatile['SKU']}) has the highest volatility "
        f"(CV {most_volatile['Coefficient_of_Variation']:.2f})",
    ]
    if promo_lift is not None and not np.isnan(promo_lift):
        insight_items.append(f"Promotion days show a <b>{promo_lift:+.1f}%</b> difference in avg units sold")
    insight_cards(insight_items, columns=3)

    group_label("Sales Performance")
    col1, col2 = st.columns(2)
    with col1:
        section("Revenue Trend Over Time")
        caption("Recent window shown by default — scroll/zoom in the chart for full history.")
        fig = px.line(weekly_agg, x="week_start", y="Revenue", template="plotly_dark",
                      labels={"week_start": "Week", "Revenue": "Revenue (₹)"}, color_discrete_sequence=["#4F7DFA"])
        recent_start = weekly_agg["week_start"].max() - pd.Timedelta(weeks=RECENT_WEEKS_DEFAULT * 2)
        fig.update_xaxes(range=[recent_start, weekly_agg["week_start"].max()])
        st.plotly_chart(dark_layout(fig, 320), use_container_width=True)
    with col2:
        section("Units Trend Over Time")
        caption("Recent window shown by default — scroll/zoom in the chart for full history.")
        fig = px.line(weekly_agg, x="week_start", y="Units_Sold", template="plotly_dark",
                      labels={"week_start": "Week", "Units_Sold": "Units Sold"}, color_discrete_sequence=["#7C6CF6"])
        fig.update_xaxes(range=[recent_start, weekly_agg["week_start"].max()])
        st.plotly_chart(dark_layout(fig, 320), use_container_width=True)

    group_label("Product Performance")
    col3, col4 = st.columns(2)
    with col3:
        section("Top 10 Products by Revenue")
        top_rev = sku_perf.sort_values("Revenue", ascending=False).head(10)
        fig = px.bar(top_rev, x="Revenue", y="Product_Name", orientation="h", template="plotly_dark",
                     labels={"Revenue": "Revenue (₹)", "Product_Name": ""}, color_discrete_sequence=["#7C6CF6"])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(dark_layout(fig, 360), use_container_width=True)
    with col4:
        section("Top 10 Products by Units Sold")
        top_units = sku_perf.sort_values("Units_Sold", ascending=False).head(10)
        fig = px.bar(top_units, x="Units_Sold", y="Product_Name", orientation="h", template="plotly_dark",
                     labels={"Units_Sold": "Units Sold", "Product_Name": ""}, color_discrete_sequence=["#4F7DFA"])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(dark_layout(fig, 360), use_container_width=True)

    group_label("Category Performance")
    col5, col6 = st.columns(2)
    cat_units = modeling_df.groupby("Category")["Units_Sold"].sum().sort_values(ascending=False)
    with col5:
        section("Revenue by Category")
        fig = px.bar(cat_rev, x=cat_rev.index, y=cat_rev.values, template="plotly_dark",
                     labels={"x": "Category", "y": "Revenue (₹)"}, color_discrete_sequence=["#22C55E"])
        st.plotly_chart(dark_layout(fig, 300), use_container_width=True)
    with col6:
        section("Units by Category")
        fig = px.bar(cat_units, x=cat_units.index, y=cat_units.values, template="plotly_dark",
                     labels={"x": "Category", "y": "Units Sold"}, color_discrete_sequence=["#22C55E"])
        st.plotly_chart(dark_layout(fig, 300), use_container_width=True)

    group_label("Demand Drivers")
    col7, col8 = st.columns(2)
    with col7:
        section("Seasonality")
        fig = px.bar(season_perf, x=season_perf.index, y=season_perf.values, template="plotly_dark",
                     labels={"x": "Season", "y": "Units Sold"}, color_discrete_sequence=["#F59E0B"])
        st.plotly_chart(dark_layout(fig, 300), use_container_width=True)
    with col8:
        section("Promotion vs Non-Promotion")
        fig = px.bar(promo_compare, x="Promotion", y="Units_Sold", template="plotly_dark",
                     labels={"Units_Sold": "Avg Units Sold / SKU-day", "Promotion": ""},
                     color_discrete_sequence=["#F59E0B"])
        st.plotly_chart(dark_layout(fig, 300), use_container_width=True)

    group_label("Demand Volatility")
    section("Volatility Distribution", "Coefficient of variation across products — higher means less predictable demand.")
    fig = px.histogram(sku_perf, x="Coefficient_of_Variation", nbins=15, template="plotly_dark",
                        labels={"Coefficient_of_Variation": "Coefficient of Variation"},
                        color_discrete_sequence=["#A78BFA"])
    st.plotly_chart(dark_layout(fig, 300), use_container_width=True)


# ===========================================================================
# PAGE — FORECAST
# ===========================================================================
def page_forecast():
    page_header("Forecast", "What are we likely to sell over the next 6 weeks?")

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Production Model", "Seasonal-Naive")
    with c2: kpi_card("WAPE", f"{baseline_wape*100:.2f}%" if baseline_wape is not None else "n/a")
    with c3: kpi_card("Forecast Horizon", "6 weeks")

    if baseline_wape is not None and ml_wape is not None:
        section("Model Comparison")
        model_comparison_cards(baseline_wape, ml_wape)

    section("SKU Forecast")
    sku_list = sorted(sku_perf["SKU"].unique())
    default_sku = sku_perf.sort_values("Revenue", ascending=False).iloc[0]["SKU"]
    sel_sku = st.selectbox("Select a SKU", sku_list, index=sku_list.index(default_sku), key="fc_sku")

    caption("Blue solid = historical actual demand · amber dashed = the production 6-week forecast. "
            "Zoom out in the chart to inspect full history.")
    st.plotly_chart(actual_vs_forecast_chart(sel_sku), use_container_width=True)

    fc_sku = forecast_df[forecast_df["SKU"] == sel_sku].sort_values("forecast_week").reset_index(drop=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        section("6-Week Forecast")
        fc_show = pd.DataFrame({
            "Week": [f"Week {i+1}" for i in range(len(fc_sku))],
            "Date": fc_sku["forecast_week"].apply(fmt_date),
            "Predicted Demand": fc_sku["predicted_demand"].round(0).astype(int),
        })
        st.dataframe(fc_show, use_container_width=True, hide_index=True)
    with col2:
        section("Total Expected Demand")
        kpi_card("Next 6 Weeks", f"{fc_sku['predicted_demand'].sum():,.0f} units")


# ===========================================================================
# PAGE — INVENTORY DASHBOARD
# ===========================================================================
def page_inventory_dashboard():
    page_header("Inventory Dashboard", "How much stock do we have relative to expected demand?")

    sel_categories, sel_skus = compact_filters(inv_enriched, key_prefix="inv")
    inv_f = inv_enriched[inv_enriched["Category"].isin(sel_categories) & inv_enriched["SKU"].isin(sel_skus)]

    if inv_f.empty:
        st.info("No SKUs match the current filters.")
        return

    fc_totals = forecast_df.groupby("SKU")["predicted_demand"].sum().rename("forward_demand_6w")
    inv_f = inv_f.merge(fc_totals, on="SKU", how="left")

    total_stock = inv_f["Current_Stock"].sum()
    total_on_order = inv_f["On_Order"].sum()
    total_inv_value = inv_f["Inventory_Value"].sum()
    avg_coverage = inv_f["Days_of_Stock_On_Hand"].mean()
    n_below_reorder = int((inv_f["Current_Stock"] < inv_f["Reorder_Point"]).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Total Current Stock", f"{total_stock:,.0f} units")
    with c2: kpi_card("Total On Order", f"{total_on_order:,.0f} units")
    with c3: kpi_card("Total Inventory Value", fmt_rs(total_inv_value))
    with c4: kpi_card("Avg Inventory Coverage", fmt_days(avg_coverage))
    with c5: kpi_card("SKUs Below Reorder Point", str(n_below_reorder), "kpi-warn" if n_below_reorder else "kpi-good")

    top_n = st.selectbox("Show", ["Top 10 SKUs", "Top 20 SKUs"], key="inv_topn")
    n = 10 if top_n == "Top 10 SKUs" else 20

    col1, col2 = st.columns(2)
    with col1:
        section("Inventory Value by Category")
        val_by_cat = inv_f.groupby("Category")["Inventory_Value"].sum().sort_values(ascending=False)
        fig = px.bar(val_by_cat, x=val_by_cat.index, y=val_by_cat.values, template="plotly_dark",
                     labels={"x": "Category", "y": "Inventory Value (₹)"}, color_discrete_sequence=["#7C6CF6"])
        st.plotly_chart(dark_layout(fig, 340), use_container_width=True)
    with col2:
        section(f"Current Stock vs 6-Week Forecast ({top_n})")
        topn_val = inv_f.sort_values("Inventory_Value", ascending=False).head(n).sort_values("Current_Stock")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=topn_val["Product_Name"], x=topn_val["Current_Stock"], name="Current Stock",
                              orientation="h", marker_color="#4F7DFA",
                              customdata=topn_val["SKU"], hovertemplate="%{y}<br>SKU: %{customdata}<br>Stock: %{x:,.0f}<extra></extra>"))
        fig.add_trace(go.Bar(y=topn_val["Product_Name"], x=topn_val["forward_demand_6w"], name="6-Week Forecast",
                              orientation="h", marker_color="#A78BFA",
                              customdata=topn_val["SKU"], hovertemplate="%{y}<br>SKU: %{customdata}<br>Forecast: %{x:,.0f}<extra></extra>"))
        fig.update_layout(barmode="group")
        st.plotly_chart(dark_layout(fig, 340), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section("Inventory Coverage Distribution")
        fig = px.histogram(inv_f, x="Days_of_Stock_On_Hand", nbins=15, template="plotly_dark",
                            labels={"Days_of_Stock_On_Hand": "Days of Stock on Hand"},
                            color_discrete_sequence=["#22C55E"])
        st.plotly_chart(dark_layout(fig, 320), use_container_width=True)
    with col4:
        section("Top Inventory-Value SKUs")
        top_val = inv_f.sort_values("Inventory_Value", ascending=False).head(10)
        fig = px.bar(top_val, x="Inventory_Value", y="Product_Name", orientation="h", template="plotly_dark",
                     labels={"Inventory_Value": "Inventory Value (₹)", "Product_Name": ""},
                     color_discrete_sequence=["#4F7DFA"])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(dark_layout(fig, 320), use_container_width=True)

    section(f"Stock vs Reorder Point ({top_n})")
    stock_vs_rp = inv_f.sort_values("Current_Stock", ascending=False).head(n).sort_values("Current_Stock")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=stock_vs_rp["Product_Name"], x=stock_vs_rp["Current_Stock"], name="Current Stock",
                          orientation="h", marker_color="#4F7DFA",
                          customdata=stock_vs_rp["SKU"], hovertemplate="%{y}<br>SKU: %{customdata}<br>Stock: %{x:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(y=stock_vs_rp["Product_Name"], x=stock_vs_rp["Reorder_Point"], name="Reorder Point",
                              mode="markers", marker=dict(color="#EF4444", size=10, symbol="diamond"),
                              customdata=stock_vs_rp["SKU"], hovertemplate="%{y}<br>SKU: %{customdata}<br>Reorder Point: %{x:,.0f}<extra></extra>"))
    fig.update_layout(height=max(340, 24 * n))
    st.plotly_chart(dark_layout(fig, max(340, 24 * n)), use_container_width=True)

    section("Inventory Health Table")
    health = inv_f.copy()
    health["Status"] = np.where(health["Current_Stock"] < health["Reorder_Point"], "⚠ Attention", "✓ Healthy")
    health["Coverage (days)"] = health["Days_of_Stock_On_Hand"].round(0).astype(int)
    health_show = health[["SKU", "Product_Name", "Category", "Current_Stock", "Reorder_Point",
                           "Coverage (days)", "Status"]].rename(
        columns={"Product_Name": "Product", "Current_Stock": "Current Stock", "Reorder_Point": "Reorder Point"})
    st.dataframe(health_show.sort_values("Status"), use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE — RISK DASHBOARD
# ===========================================================================
def page_risk_dashboard():
    page_header("Risk Dashboard", "What should the operations team do?")

    counts = risk_df["risk_action"].value_counts().reindex(ACTION_ORDER).fillna(0).astype(int)
    c1, c2, c3, c4 = st.columns(4)
    for col, action in zip([c1, c2, c3, c4], ACTION_ORDER):
        with col:
            kpi_card(action, str(counts[action]), ACTION_CSS[action])

    st.markdown('<div class="insight-box">The risk engine combines the production forecast with the latest '
                'inventory position to prioritize action.</div>', unsafe_allow_html=True)

    section("Stockout vs Overstock Decision Matrix")
    fig = quadrant_decision_matrix(risk_df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    section("Prioritized Action Table")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        action_choice = st.selectbox("Action", ["All Actions"] + ACTION_ORDER, key="rd_action")
    sel_actions = ACTION_ORDER if action_choice == "All Actions" else [action_choice]
    with fcol2:
        cat_choice = st.selectbox("Category", ["All Categories"] + sorted(risk_df["Category"].unique()), key="rd_cat")
    sel_cats = sorted(risk_df["Category"].unique()) if cat_choice == "All Categories" else [cat_choice]
    filtered_pre = risk_df[risk_df["risk_action"].isin(sel_actions) & risk_df["Category"].isin(sel_cats)]
    with fcol3:
        sku_choice = st.selectbox("SKU", ["All SKUs"] + sorted(filtered_pre["SKU"].unique()), key="rd_sku")
    filtered = filtered_pre if sku_choice == "All SKUs" else filtered_pre[filtered_pre["SKU"] == sku_choice]
    filtered = filtered.sort_values("priority_score", ascending=False)

    if filtered.empty:
        st.info("No SKUs match the current filters.")
    else:
        render_table(priority_table_html(filtered))

    section("Top Reorder Candidates")
    top_reorder = risk_df[risk_df["risk_action"] == "REORDER NOW"].sort_values("priority_score", ascending=False).head(5)
    if top_reorder.empty:
        st.info("No SKUs currently flagged REORDER NOW.")
    else:
        render_table(priority_table_html(top_reorder))

    section("Top Markdown / Clear Candidates")
    top_markdown = risk_df[risk_df["risk_action"] == "MARKDOWN / CLEAR"].sort_values("priority_score", ascending=False).head(5)
    if top_markdown.empty:
        st.info("No SKUs currently flagged MARKDOWN / CLEAR.")
    else:
        render_table(priority_table_html(top_markdown))

    section("SKU Drilldown")
    drill_options = sorted(filtered["SKU"].unique()) if not filtered.empty else sorted(risk_df["SKU"].unique())
    sel_drill = st.selectbox("Select a SKU to inspect", drill_options, key="rd_drill")
    row = risk_df[risk_df["SKU"] == sel_drill].iloc[0]
    status_banner(row["risk_action"])
    render_sku_drilldown(row)


def render_sku_drilldown(row: pd.Series):
    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    with dcol1:
        st.markdown('<div class="subsection-label">Current Position</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card" style="min-height:auto; white-space:normal;">
        <b>{row['Product_Name']}</b> ({row['SKU']}) — {row['Category']}<br><br>
        Current Stock: <b>{fmt_units(row['Current_Stock'])}</b><br>
        On Order: <b>{fmt_units(row['On_Order'])}</b><br>
        Safety Stock: <b>{fmt_units(row['Safety_Stock'])}</b><br>
        Reorder Point: <b>{fmt_units(row['Reorder_Point'])}</b><br>
        Lead Time: <b>{fmt_days(row['Lead_Time_Days'])}</b>
        </div>""", unsafe_allow_html=True)
    with dcol2:
        st.markdown('<div class="subsection-label">Forecast</div>', unsafe_allow_html=True)
        coverage = f"{row['inventory_coverage_weeks']:.1f} weeks" if pd.notna(row['inventory_coverage_weeks']) else "n/a"
        st.markdown(f"""
        <div class="kpi-card" style="min-height:auto; white-space:normal;">
        6-Week Forecast: <b>{fmt_units(row['forward_demand_6w'])} units</b><br>
        Lead-Time Demand: <b>{fmt_units(row['lead_time_demand'])} units</b><br>
        Projected Stock After Lead Time: <b>{fmt_units(row['projected_stock_after_lead'])}</b><br>
        Inventory Coverage: <b>{coverage}</b>
        </div>""", unsafe_allow_html=True)
    with dcol3:
        st.markdown('<div class="subsection-label">Risk</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card" style="min-height:auto; white-space:normal;">
        Stockout Risk: <b>{"Yes" if row['stockout_risk'] else "No"}</b><br>
        Overstock Risk: <b>{"Yes" if row['overstock_risk'] else "No"}</b><br>
        Sales at Risk: <b>{fmt_rs(row['sales_at_risk_rs'])}</b><br>
        Capital Locked: <b>{fmt_rs(row['capital_locked_rs'])}</b><br>
        Priority Score: <b>{row['priority_score']:.2f}</b>
        </div>""", unsafe_allow_html=True)
    with dcol4:
        st.markdown('<div class="subsection-label">Recommended Action</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card" style="min-height:auto; align-items:flex-start; white-space:normal;">
        {action_pill_html(row['risk_action'])}
        <div style="margin-top:0.7rem; color:var(--muted); font-size:0.83rem;">
        Based on the production forecast and latest inventory position.
        </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="why-box"><span class="why-label">Why?</span>{why_flagged_text(row)}</div>',
                unsafe_allow_html=True)


# ===========================================================================
# PAGE — PRODUCT DETAILS
# ===========================================================================
def page_product_details():
    page_header("Product Details", "A 360° view of a single SKU.")

    sku_list = sorted(sku_perf["SKU"].unique())
    default_sku = sku_perf.sort_values("Revenue", ascending=False).iloc[0]["SKU"]
    sel_sku = st.selectbox("Select a SKU", sku_list, index=sku_list.index(default_sku), key="pd_sku")

    sp = sku_perf[sku_perf["SKU"] == sel_sku].iloc[0]
    r = risk_df[risk_df["SKU"] == sel_sku].iloc[0]
    fc = forecast_df[forecast_df["SKU"] == sel_sku]
    m = modeling_df[modeling_df["SKU"] == sel_sku]

    st.markdown(f"""
    <div class="snapshot-box">
        <div class="snapshot-title">Product Overview</div>
        <div class="snapshot-line">
            <b>{sp['Product_Name']}</b> &nbsp;·&nbsp; SKU: <span class="snapshot-num">{sel_sku}</span>
            &nbsp;·&nbsp; Category: <span class="snapshot-num">{r['Category']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    status_banner(r["risk_action"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="subsection-label">Sales</div>', unsafe_allow_html=True)
        avg_demand = m["Units_Sold"].mean()
        st.markdown(f"""
        <div class="kpi-card" style="min-height:auto; white-space:normal;">
        Total Units Sold: <b>{fmt_units(sp['Units_Sold'])}</b><br>
        Total Revenue: <b>{fmt_rs(sp['Revenue'])}</b><br>
        Avg Daily Demand: <b>{avg_demand:.1f}</b><br>
        Volatility (CV): <b>{sp['Coefficient_of_Variation']:.2f}</b>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="subsection-label">Forecast</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card" style="min-height:auto; white-space:normal;">
        6-Week Demand: <b>{fmt_units(fc['predicted_demand'].sum())} units</b><br>
        Production Model: <b>Seasonal-Naive</b><br>
        WAPE: <b>{f"{baseline_wape*100:.2f}%" if baseline_wape is not None else "n/a"}</b>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="subsection-label">Inventory</div>', unsafe_allow_html=True)
        coverage = f"{r['inventory_coverage_weeks']:.1f} weeks" if pd.notna(r['inventory_coverage_weeks']) else "n/a"
        st.markdown(f"""
        <div class="kpi-card" style="min-height:auto; white-space:normal;">
        Current Stock: <b>{fmt_units(r['Current_Stock'])}</b><br>
        On Order: <b>{fmt_units(r['On_Order'])}</b><br>
        Safety Stock: <b>{fmt_units(r['Safety_Stock'])}</b> · Reorder Point: <b>{fmt_units(r['Reorder_Point'])}</b><br>
        Lead Time: <b>{fmt_days(r['Lead_Time_Days'])}</b> · Coverage: <b>{coverage}</b>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="subsection-label">Risk</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpi-card" style="min-height:auto; white-space:normal;">
        {action_pill_html(r['risk_action'])}<br><br>
        Sales at Risk: <b>{fmt_rs(r['sales_at_risk_rs'])}</b><br>
        Capital Locked: <b>{fmt_rs(r['capital_locked_rs'])}</b><br>
        Priority Score: <b>{r['priority_score']:.2f}</b>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="why-box"><span class="why-label">Why This Product Is Flagged</span>'
                f'{why_flagged_text(r)}</div>', unsafe_allow_html=True)

    section("Actual vs Forecast")
    st.plotly_chart(actual_vs_forecast_chart(sel_sku), use_container_width=True)


# ===========================================================================
# PAGE — EXECUTIVE SUMMARY
# ===========================================================================
def page_executive_summary():
    st.markdown("""
    <div class="foresight-header">
        <p class="foresight-title">FORESIGHT</p>
        <p class="foresight-sub">Executive Decision Summary</p>
    </div>
    """, unsafe_allow_html=True)

    total_revenue = modeling_df["Revenue"].sum()
    sales_at_risk = risk_df["sales_at_risk_rs"].sum()
    capital_locked = risk_df["capital_locked_rs"].sum()
    n_action = int((risk_df["risk_action"] != "HEALTHY").sum())
    n_reorder = int((risk_df["risk_action"] == "REORDER NOW").sum())
    n_markdown = int((risk_df["risk_action"] == "MARKDOWN / CLEAR").sum())
    n_healthy = int((risk_df["risk_action"] == "HEALTHY").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Revenue", fmt_rs(total_revenue))
    with c2: kpi_card("Sales at Risk", fmt_rs(sales_at_risk), "kpi-danger")
    with c3: kpi_card("Capital Locked", fmt_rs(capital_locked), "kpi-warn")
    with c4: kpi_card("Action Required", f"{n_action} SKUs", "kpi-warn" if n_action else "kpi-good")

    d1, d2, d3 = st.columns(3)
    with d1: kpi_card("REORDER NOW", str(n_reorder), "kpi-danger")
    with d2: kpi_card("MARKDOWN / CLEAR", str(n_markdown), "kpi-accent")
    with d3: kpi_card("HEALTHY", str(n_healthy), "kpi-good")

    section("Business Situation")
    st.markdown(f"""
    <div class="insight-box">
    Across {modeling_df['SKU'].nunique()} modeled SKUs, {fmt_rs(total_revenue)} in revenue has been generated
    to date. {n_action} of {len(risk_df)} SKUs currently require operational attention:
    {n_reorder} are at risk of stocking out and {n_markdown} are overstocked relative to expected demand,
    together representing an estimated {fmt_rs(sales_at_risk)} in sales exposure and
    {fmt_rs(capital_locked)} in locked capital.
    </div>
    """, unsafe_allow_html=True)

    section("Key Findings")
    cat_rev = modeling_df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    season_perf = modeling_df.groupby("season")["Units_Sold"].sum().sort_values(ascending=False)
    top_sku_row = sku_perf.sort_values("Revenue", ascending=False).iloc[0]
    most_volatile = sku_perf.sort_values("Coefficient_of_Variation", ascending=False).iloc[0]
    n_negative_margin = int((sku_perf["Gross_Margin_Per_Unit"] < 0).sum())
    findings = [
        f"<b>{cat_rev.index[0]}</b> is the strongest category by revenue ({fmt_rs(cat_rev.iloc[0])})",
        f"<b>{season_perf.index[0]}</b> is the peak demand season historically",
        f"<b>{top_sku_row['Product_Name']}</b> ({top_sku_row['SKU']}) is the top revenue SKU "
        f"({fmt_rs(top_sku_row['Revenue'])})",
        f"<b>{most_volatile['Product_Name']}</b> ({most_volatile['SKU']}) has the highest volatility "
        f"(CV {most_volatile['Coefficient_of_Variation']:.2f})",
        f"<b>{n_negative_margin}</b> of {len(sku_perf)} SKUs carry a negative gross margin per unit",
        f"The risk engine flags <b>{n_reorder}</b> SKUs for reorder and <b>{n_markdown}</b> for markdown/clearance",
    ]
    insight_cards(findings, columns=3)

    section("Decisions Required")
    insight_cards([
        f"<b>{n_reorder} SKUs</b> require replenishment review (REORDER NOW)",
        f"<b>{n_markdown} SKUs</b> require markdown/clearance review (MARKDOWN / CLEAR)",
        f"<b>{fmt_rs(sales_at_risk)}</b> estimated sales exposure from potential stockouts",
        f"<b>{fmt_rs(capital_locked)}</b> estimated excess capital exposure from overstock",
    ], columns=4)

    if baseline_wape is not None and ml_wape is not None:
        section("Forecast / Model Performance")
        model_comparison_cards(baseline_wape, ml_wape)

    section("Top 5 Highest-Priority SKUs")
    top5 = risk_df.sort_values("priority_score", ascending=False).head(5)
    render_table(priority_table_html(top5))

    section("Final Recommendation")
    st.markdown(f"""
    <div class="why-box">
    <span class="why-label">Recommendation</span>
    Focus first on the {n_reorder} REORDER NOW candidates to avoid lost sales, while reducing excess stock
    in the {n_markdown} MARKDOWN / CLEAR candidates to free up locked capital.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
ROUTES = {
    "🏠 Home": page_home,
    "📊 Sales Analytics": page_sales_analytics,
    "🔮 Forecast": page_forecast,
    "📦 Inventory Dashboard": page_inventory_dashboard,
    "⚠️ Risk Dashboard": page_risk_dashboard,
    "🔎 Product Details": page_product_details,
    "🎯 Executive Summary": page_executive_summary,
}
ROUTES[page]()
