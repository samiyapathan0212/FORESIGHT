# FORESIGHT — Demand & Inventory Intelligence

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.6x-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.14x-009688?logo=fastapi&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)

An end-to-end demand forecasting and inventory risk system that turns raw sales and stock data into prioritized reorder and markdown decisions, delivered through an interactive Streamlit dashboard.

---

## Problem Statement

A direct-to-consumer home & lifestyle brand plans inventory manually, using spreadsheets and judgement. This creates losses in two directions at once:

- **Stockouts** — fast-moving products run out, and those sales are never recovered.
- **Overstock** — slow movers accumulate, locking up working capital and eventually requiring markdowns.

The operations team needs to answer three questions every planning cycle:

1. How much are we likely to sell over the next few weeks?
2. Which products are about to run out?
3. Which products are we holding far more of than we will sell?

FORESIGHT addresses all three by combining a weekly SKU-level demand forecast with the latest inventory position, then converting the result into a ranked, explainable action list.

---

## Solution Overview

```
Raw Data → Data Validation & Cleaning → EDA → Forecasting → Risk Engine → Prioritized Actions → Streamlit Dashboard
```

| Stage | What happens | Artifact |
|---|---|---|
| Validation & cleaning | Schema checks, type/date parsing, duplicate and null checks, SKU relationship validation | `src/pipeline.py` → `data/processed/`, `reports/data_quality_report.*` |
| EDA | Demand patterns, seasonality, category and promotion analysis, margin and volatility profiling | `notebooks/01_eda.ipynb` → `reports/eda_key_findings.txt` |
| Baseline | Weekly seasonal-naive forecast with rolling-origin backtesting | `notebooks/02_baseline.ipynb` → `reports/baseline_summary.txt` |
| Model comparison | Global gradient-boosting model backtested against the baseline | `notebooks/03_model.ipynb` → `reports/forecast_metrics.csv` |
| Risk engine | Stockout/overstock scoring, rupee exposure, priority ranking | `src/risk.py` → `data/processed/risk_output.csv` |
| Dashboard | Seven-page interactive analytics application | `app/app.py` |

The pipeline is reproducible from raw data with a single command, and the dashboard consumes generated outputs rather than recalculating any forecasting or risk logic.

---

## Key Capabilities

- **Historical sales analysis** — revenue and unit trends, category and product performance, seasonality and promotion comparison
- **Demand forecasting** — weekly, SKU-level, 6-week horizon
- **Model comparison** — seasonal-naive baseline vs. gradient-boosting, evaluated on identical rolling-origin backtest windows
- **Inventory analysis** — stock position, coverage, inventory value, reorder-point health
- **Stockout detection** — lead-time demand compared against on-hand plus on-order supply
- **Overstock detection** — current stock compared against forward forecast demand
- **Priority scoring** — transparent additive score combining stockout severity, overstock severity and rupee exposure
- **Reorder recommendations** — ranked REORDER NOW candidates
- **Markdown/clear recommendations** — ranked MARKDOWN / CLEAR candidates
- **SKU-level drilldown** — 360° view of a single product across sales, forecast, inventory and risk
- **Executive dashboard** — presentation-ready decision summary

---

## Data

### Raw inputs (`data/`)

| File | Rows | Grain | Purpose |
|---|---|---|---|
| `sales_daily.csv` | 36,550 | One row per SKU per day | Units sold, revenue, price, promotion flag |
| `sku_master.csv` | 50 | One row per SKU | Category, subcategory, launch date, cost and selling price, gross margin |
| `calendar.csv` | 731 | One row per date | Week, month, quarter, season, holiday, promotion event |
| `inventory_snapshots.csv` | 4,800 | Monthly stock position per SKU | On-hand, on-order, lead time, safety stock, reorder point, inventory value |

Date coverage: **2024-01-01 to 2025-12-31**. Inventory snapshots run monthly to **2025-12-01**.

### Processed outputs (`data/processed/`)

| File | Purpose |
|---|---|
| `sales_clean.csv`, `sku_master_clean.csv`, `calendar_clean.csv`, `inventory_clean.csv` | Validated and type-corrected source tables |
| `modeling_data.csv` | Analysis-ready joined dataset (36,550 rows × 29 columns, 50 SKUs) |
| `weekly_demand_panel.csv` | Weekly SKU demand panel with the seasonal-naive baseline attached |
| `sku_performance_summary.csv` | Per-SKU revenue, units, volatility and margin |
| `inventory_latest_snapshot_summary.csv` | Latest inventory snapshot per SKU with coverage |
| `baseline_backtest_results.csv` | Per-window baseline backtest metrics |
| `forecast_output.csv` | Production 6-week forecast per SKU |
| `risk_output.csv` | Final risk classification and rupee exposure per SKU |

### Documented data-quality finding

Inventory tracks **200 SKUs**, but sales and SKU master cover only **SKU001–SKU050**. The **150 inventory-only SKUs** are retained in the cleaned inventory file but excluded from forecasting and risk decisions, since no sales or product-master history exists for them. This is reported explicitly in `reports/data_quality_report.txt` rather than silently dropped.

---

## Machine Learning / Forecasting

**Framing:** weekly SKU-level demand, 6-week forecast horizon, evaluated with **rolling-origin (walk-forward) backtesting** across 8 non-overlapping windows. The primary metric is **WAPE** (total absolute error ÷ total actual demand), with **bias** as a secondary check.

### Models evaluated

| Model | Description | WAPE | Bias |
|---|---|---|---|
| **Seasonal-Naive (production)** | Demand equals the value observed 52 weeks earlier for the same SKU | **0.1180 (11.80%)** | +0.0136 |
| HistGradientBoosting (global) | Single global model across all 50 SKUs; direct multi-horizon strategy with lag, rolling, calendar, promotion and SKU/category features | 0.1520 (15.20%) | +0.0832 |

### Result

The gradient-boosting model scored **28.8% worse** than the baseline on the backtest. Per the project's methodology, the baseline was retained rather than reporting a fabricated improvement:

> **The production forecast is the Seasonal-Naive baseline.** The machine-learning model is documented as an evaluated-and-rejected alternative and is not used for any forecast or risk decision.

### Leakage controls

- All lag and rolling features are shifted so they use past information only.
- Each backtest origin trains only on targets already observed at that origin.
- Time-based windows are used throughout; no random train/test split.

---

## Risk Engine

Implemented in `src/risk.py`. The logic is deterministic business rules, not a model — every threshold is a named, configurable constant.

### Stockout risk

```
lead_time_weeks            = ceil(Lead_Time_Days / 7), capped at the forecast horizon
lead_time_demand           = forecast demand summed over the lead-time window
available_supply           = Current_Stock + On_Order
projected_stock_after_lead = available_supply - lead_time_demand

stockout_risk         = projected_stock_after_lead < Safety_Stock
estimated_units_short = max(Safety_Stock - projected_stock_after_lead, 0)
```

### Overstock risk

```
forward_demand_6w = forecast demand summed over the 6-week horizon
overstock_risk    = Current_Stock > forward_demand_6w × OVERSTOCK_MULTIPLIER   (default 2.0)
excess_inventory  = max(Current_Stock - forward_demand_6w × OVERSTOCK_MULTIPLIER, 0)
```

If forward demand is zero, any positive stock is treated as overstock and the full stock counts as excess. Inventory coverage is left undefined rather than dividing by zero.

### Rupee exposure

```
sales_at_risk_rs      = estimated_units_short × Selling_Price
capital_locked_rs     = excess_inventory      × Cost_Price
total_rupee_impact_rs = sales_at_risk_rs + capital_locked_rs
```

These are labelled **estimated exposure** based on documented assumptions, not confirmed financial losses.

### Priority score

A transparent additive score so the dashboard can rank the most urgent products first:

```
stockout_severity  = estimated_units_short / (Safety_Stock + 1)
overstock_severity = excess_inventory      / (forward_demand_6w + 1)
rupee_severity     = total_rupee_impact_rs / max(total_rupee_impact_rs)

priority_score = stockout_severity + overstock_severity + rupee_severity
```

### Action classification

| Stockout risk | Overstock risk | Action | Meaning |
|---|---|---|---|
| False | False | **HEALTHY** | No action needed |
| True | False | **REORDER NOW** | Replenish before stock runs out |
| False | True | **MARKDOWN / CLEAR** | Promote or discount to free capital |
| True | True | **WATCH / VOLATILE** | Review manually |

Data-quality assertions enforce exactly one inventory record and one final risk row per modeled SKU, with no duplicates and no future inventory information.

---

## Dashboard

A seven-page Streamlit application (`app/app.py`) with a dark analytics theme and interactive Plotly visuals. All values are read from generated outputs.

| Page | Business question it answers |
|---|---|
| **Home** | What is FORESIGHT and what is the current business situation? Headline KPIs, management snapshot, pipeline flow |
| **Sales Analytics** | What has happened historically? Revenue and unit trends, product and category performance, seasonality, promotion comparison, volatility, key insights |
| **Forecast** | What are we likely to sell? Model summary, baseline vs. ML comparison, actual-vs-forecast chart with a forecast-start marker, 6-week forecast table |
| **Inventory Dashboard** | How much stock do we hold relative to expected demand? Stock, on-order, inventory value, coverage, reorder-point health |
| **Risk Dashboard** | What should the operations team do? Four-quadrant decision matrix, prioritized action table, top reorder and markdown candidates, SKU drilldown |
| **Product Details** | A 360° view of one SKU across sales, forecast, inventory and risk, with a plain-language explanation of why it is flagged |
| **Executive Summary** | Presentation-ready readout: business situation, key findings, decisions required, model performance, top-5 priorities, final recommendation |

---

## Key Results

All figures below are generated by the project and stored in `reports/` and `data/processed/`.

### Data foundation

| Metric | Value |
|---|---|
| Modeled SKUs (sales + master) | 50 |
| Inventory-only SKUs excluded from modeling | 150 |
| Modeling dataset | 36,550 rows × 29 columns |
| Date coverage | 2024-01-01 to 2025-12-31 |
| Duplicates / nulls after cleaning | 0 |

### Forecast performance

| Metric | Value |
|---|---|
| Production model | Seasonal-Naive baseline |
| Forecast horizon | 6 weeks |
| Backtest windows | 8 (rolling-origin, non-overlapping) |
| Baseline WAPE | **11.80%** (bias +0.0136) |
| HistGradientBoosting WAPE | 15.20% (bias +0.0832) |
| Model vs. baseline | −28.8% (model did not beat baseline) |

### Inventory risk (latest snapshot: 2025-12-01)

| Action | SKUs |
|---|---|
| REORDER NOW | 8 |
| MARKDOWN / CLEAR | 2 |
| WATCH / VOLATILE | 0 |
| HEALTHY | 40 |

| Exposure | Estimated value |
|---|---|
| Sales at risk (stockouts) | ₹65.24 L |
| Capital locked (overstock) | ₹21.11 L |
| Total estimated rupee impact | ₹86.35 L |

### Selected EDA findings

- Home Decor is the strongest category by revenue; Lighting is the weakest.
- Revenue is concentrated: 24 of 50 SKUs (~48%) account for roughly 80% of total revenue.
- Winter is the strongest season overall; March 2024 was the peak demand month.
- Promotion days show a **+38.1%** difference in average units sold per SKU-day versus non-promotion days (observed difference, not proof of causation).
- **16 of 50 SKUs (32%)** carry a negative gross margin per unit.
- Correlation between total units sold and gross margin per unit is only **0.14** — high sales volume does not imply high profitability.
- Demand volatility (coefficient of variation) ranges from **0.25 to 0.65** across SKUs.

---

## Project Architecture

```
                    Data Sources
      sales_daily · sku_master · calendar · inventory_snapshots
                          ↓
                Validation & Cleaning
         schema · types · duplicates · SKU relationships
                    (src/pipeline.py)
                          ↓
                          EDA
        demand patterns · seasonality · margin · volatility
                  (notebooks/01_eda.ipynb)
                          ↓
                      Forecasting
     seasonal-naive baseline vs. gradient boosting (backtested)
             (notebooks/02_baseline · 03_model)
                          ↓
                      Risk Engine
      stockout · overstock · rupee exposure · priority score
                      (src/risk.py)
                          ↓
                    Decision Layer
     REORDER NOW · MARKDOWN / CLEAR · WATCH / VOLATILE · HEALTHY
                          ↓
                  Streamlit Dashboard
                      (app/app.py)
```

---

## Repository Structure

```
ForeSight/
├── app/
│   ├── app.py                              # Streamlit dashboard (7 pages)
│   └── requirements.txt                    # Dashboard dependencies
├── data/
│   ├── sales_daily.csv                     # Raw inputs
│   ├── sku_master.csv
│   ├── calendar.csv
│   ├── inventory_snapshots.csv
│   └── processed/
│       ├── sales_clean.csv
│       ├── sku_master_clean.csv
│       ├── calendar_clean.csv
│       ├── inventory_clean.csv
│       ├── modeling_data.csv               # Analysis-ready dataset
│       ├── weekly_demand_panel.csv
│       ├── sku_performance_summary.csv
│       ├── inventory_latest_snapshot_summary.csv
│       ├── baseline_backtest_results.csv
│       ├── forecast_output.csv             # Production forecast
│       └── risk_output.csv                 # Risk classification per SKU
├── notebooks/
│   ├── 01_eda.ipynb                        # Exploratory data analysis
│   ├── 02_baseline.ipynb                   # Seasonal-naive baseline + backtest
│   ├── 03_model.ipynb                      # Model comparison
│   └── 04_risk.ipynb                       # Risk engine analysis
├── reports/
│   ├── data_quality_report.txt / .json
│   ├── eda_key_findings.txt
│   ├── baseline_summary.txt
│   ├── forecast_summary.txt
│   ├── forecast_metrics.csv
│   ├── risk_summary.txt
│   └── risk_metrics.csv
├── service/
│   ├── main.py                             # FastAPI scoring service (D6)
│   └── requirements.txt                    # API dependencies
├── src/
│   ├── pipeline.py                         # Ingest, validate, clean, join
│   └── risk.py                             # Stockout / overstock risk engine
└── README.md
```

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.12 |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn (`HistGradientBoostingRegressor`) |
| Visualization | Plotly, matplotlib, seaborn |
| Dashboard | Streamlit |
| API | FastAPI, Uvicorn |
| Exploration | Jupyter Notebooks |

---

## How to Run

Commands below are for **Windows** (Command Prompt), run from the project root.

**1. Create and activate the virtual environment** (skip creation if `venv\` already exists)

```bat
python -m venv venv
venv\Scripts\activate
```

**2. Install dependencies**

```bat
pip install -r app\requirements.txt
pip install scikit-learn matplotlib seaborn jupyter
```

**3. Run the data pipeline** (validates, cleans and rebuilds `data/processed/` and the data-quality report)

```bat
python src\pipeline.py
```

**4. Run the risk engine** (regenerates `risk_output.csv` and the risk reports)

```bat
python src\risk.py
```

**5. Launch the dashboard**

```bat
streamlit run app\app.py
```

The dashboard opens at `http://localhost:8501`.

**Optional — open the notebooks**

```bat
jupyter notebook
```

> The notebooks (`02_baseline.ipynb` → `03_model.ipynb`) regenerate `weekly_demand_panel.csv`, `forecast_output.csv` and the forecast reports. Run them in order if you want to rebuild the forecast from scratch; otherwise the committed outputs are sufficient for the dashboard.

---

## Scoring Service (API)

A read-only FastAPI service (`service/main.py`) exposes the production forecast and risk assessment for any modeled SKU. It serves the generated artifacts — it does **not** recompute forecasting or risk logic, so API values always match `forecast_output.csv` and `risk_output.csv`.

### Run locally

```bat
venv\Scripts\activate
pip install -r service\requirements.txt
uvicorn service.main:app --reload --port 8000
```

Interactive documentation: `http://localhost:8000/docs`

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service status, SKU count, production model, data-source availability |
| GET | `/score/{sku}` | Forecast and risk for a single SKU |
| POST | `/score` | Forecast and risk for a batch of SKUs (1–200) |
| GET | `/docs` | Swagger UI |

SKU lookup is case-insensitive. An unknown SKU returns **404**; a malformed request body returns **422** with field-level detail.

### `GET /health`

```json
{
  "status": "ok",
  "scored_skus": 50,
  "forecast_horizon_weeks": 6,
  "production_model": "Seasonal-Naive Baseline",
  "data_sources": {
    "risk_output.csv": true,
    "forecast_output.csv": true,
    "forecast_metrics.csv": true
  }
}
```

### `GET /score/{sku}`

Request:

```bat
curl http://localhost:8000/score/SKU012
```

Response (`200`):

```json
{
  "SKU": "SKU012",
  "Product_Name": "Product 012",
  "Category": "Home Decor",
  "forecast_horizon_weeks": 6,
  "forecast_model": "Seasonal-Naive Baseline",
  "forecast_demand_total": 1135.0,
  "forecast_weekly": [
    { "week": "2026-01-05", "predicted_demand": 180.0 },
    { "week": "2026-01-12", "predicted_demand": 194.0 },
    { "week": "2026-01-19", "predicted_demand": 191.0 },
    { "week": "2026-01-26", "predicted_demand": 172.0 },
    { "week": "2026-02-02", "predicted_demand": 195.0 },
    { "week": "2026-02-09", "predicted_demand": 203.0 }
  ],
  "stockout_risk": true,
  "overstock_risk": false,
  "risk_action": "REORDER NOW",
  "sales_at_risk_rs": 2484561.71,
  "capital_locked_rs": 0.0,
  "total_rupee_impact_rs": 2484561.71,
  "priority_score": 17.647058823529413
}
```

Unknown SKU (`404`):

```json
{
  "detail": "SKU 'SKU999' not found. Only the 50 modeled SKUs (with both sales and product-master history) can be scored."
}
```

### `POST /score`

Request:

```bat
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d "{\"skus\":[\"SKU012\",\"SKU025\",\"SKU999\"]}"
```

Response (`200`) — unknown SKUs are reported rather than failing the batch:

```json
{
  "requested": 3,
  "found": 2,
  "not_found": ["SKU999"],
  "results": [
    { "SKU": "SKU012", "risk_action": "REORDER NOW", "forecast_demand_total": 1135.0, "...": "..." },
    { "SKU": "SKU025", "risk_action": "MARKDOWN / CLEAR", "forecast_demand_total": 115.0, "...": "..." }
  ]
}
```

Malformed body (`422`):

```json
{
  "detail": [
    { "type": "missing", "loc": ["body", "skus"], "msg": "Field required", "input": { "bad": 1 } }
  ]
}
```

> Only the 50 modeled SKUs are scoreable. The 150 inventory-only SKUs have no sales or product-master history and are intentionally excluded, consistent with the pipeline's documented data-quality finding.

---

## Business Value

FORESIGHT converts raw sales and inventory extracts into decisions the operations team can act on without a data scientist in the room:

- **Focused attention.** Instead of reviewing all 50 SKUs, the team sees the 10 that currently need action, ranked by priority.
- **Two-sided risk visibility.** Stockout and overstock exposure are surfaced together, so replenishment and clearance decisions are made from the same view.
- **Quantified trade-offs.** Each flagged SKU carries an estimated rupee exposure, making it possible to prioritize by materiality rather than by intuition.
- **Explainability.** Every flag comes with a plain-language reason derived from the underlying rule, so recommendations can be challenged and audited.
- **Reproducibility.** The whole pipeline re-runs from raw data with one command, so the analysis can be refreshed as new extracts arrive.
- **Honest accuracy reporting.** The forecast that is actually used is the one that performed best on a fair backtest, and the rejected alternative is documented alongside it.

Rupee figures are presented as estimated exposure under documented business-rule assumptions, not as realized or guaranteed financial outcomes.

---

## Limitations

- **Short history.** Two years of data (2024–2025) provides only a single full year-over-year cycle for the 52-week seasonal lookback. The first 52 weeks of each SKU's history fall back to a previous-week value (2,600 rows).
- **The ML model did not beat the baseline.** The gradient-boosting model scored 15.20% WAPE against the baseline's 11.80%, so the simpler method is in production. Additional history or better-engineered features may change this.
- **Inventory-only SKUs excluded.** 150 of the 200 SKUs in the inventory file have no sales or master data and are therefore outside all forecasting and risk decisions.
- **Monthly inventory granularity.** Inventory snapshots are monthly, so the risk engine uses the latest available snapshot (2025-12-01) rather than a live stock position.
- **Lead-time capping.** Lead-time demand is capped at the 6-week forecast horizon, which understates demand for SKUs whose lead time exceeds that window.
- **Business-rule thresholds.** The overstock multiplier (2.0) and safety-stock logic are configurable assumptions, not values learned from data.
- **Point forecasts only.** No prediction intervals are produced, so forecast uncertainty is not quantified per SKU.
- **Synthetic data.** The datasets are simulated extracts modelled on a real D2C brand, so results demonstrate the method rather than describe an actual business.

---

## Future Improvements

- **Stronger forecasting models** — per-SKU seasonal decomposition, hierarchical/grouped forecasting, or classical time-series methods to try to beat the seasonal-naive baseline honestly
- **Probabilistic forecasts** — prediction intervals and service-level-driven safety stock instead of point forecasts
- **Automated retraining** — scheduled re-runs with backtest monitoring so the production model is re-selected as new data arrives
- **Model monitoring** — drift and accuracy tracking to detect forecast degradation before it affects decisions
- **Deployment** — hosted dashboard plus a scoring API returning forecast and risk for a given SKU or batch
- **Database integration** — replace CSV extracts with a warehouse connection for incremental, scheduled refreshes
- **Richer inventory data** — more frequent snapshots and supplier-level lead-time variability for tighter stockout detection
- **Margin-aware prioritization** — incorporate the 16 negative-margin SKUs into the decision logic alongside volume and exposure

---

## Author

**Samiya Pathan**

Built as a data science engagement project: reproducible pipeline, honest model evaluation, transparent risk logic, and a stakeholder-facing dashboard.
