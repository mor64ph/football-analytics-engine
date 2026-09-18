# Transfer Market Predictor — PowerBI Dashboard Spec

## Data Sources

Connect PowerBI to Databricks via the native connector:
- **Home → Get Data → Azure Databricks**
- Server: `your-workspace.azuredatabricks.net`
- HTTP Path: your cluster's HTTP path
- Import these tables:
  - `transfer_market.GOLD_PLAYER_PREDICTIONS`
  - `transfer_market.GOLD_FEATURE_IMPORTANCE`

For local prototyping: load `data/processed/gold_player_predictions.csv` and `gold_feature_importance.csv` instead.

---

## Page 1 — Market Overview

**Purpose:** High-level snapshot of how well the model predicts market values across leagues.

### Visuals

| Visual | Type | Fields |
|--------|------|--------|
| Average Absolute % Error | KPI card | Mean of `abs_pct_error` |
| Model R² | KPI card | Computed measure (see DAX below) |
| Accuracy Band Distribution | Donut chart | `accuracy_band` (count) |
| Predicted vs Actual (scatter) | Scatter chart | X=`market_value_eur`, Y=`predicted_value_eur`, tooltip=`player_name` |
| Error by League | Clustered bar | X=`competition`, Y=Mean `abs_pct_error` |

### DAX Measures

```dax
Avg Abs % Error = AVERAGE(GOLD_PLAYER_PREDICTIONS[abs_pct_error])

Within 25% = 
DIVIDE(
    COUNTROWS(FILTER(GOLD_PLAYER_PREDICTIONS, GOLD_PLAYER_PREDICTIONS[abs_pct_error] <= 25)),
    COUNTROWS(GOLD_PLAYER_PREDICTIONS)
) * 100

-- Formatted label
Model Accuracy Label = FORMAT([Within 25%], "0.0") & "% within 25%"
```

---

## Page 2 — Player Deep Dive

**Purpose:** Search any player and compare their stats, predicted value, and actual value.

### Visuals

| Visual | Type | Fields |
|--------|------|--------|
| Player Search | Slicer | `player_name` (search mode) |
| League / Season filters | Slicer | `competition`, `season` |
| Player card | Multi-row card | `player_name`, `team_name`, `age`, `position`, `goals`, `assists` |
| Actual vs Predicted | 100% stacked bar (single player) | `market_value_eur` vs `predicted_value_eur` |
| % Error | KPI card (conditional format) | `pct_error` — green if <25%, amber <50%, red ≥50% |
| Top 10 Most Overvalued | Table | `player_name`, `market_value_eur`, `predicted_value_eur`, `pct_error` sorted desc |
| Top 10 Most Undervalued | Table | Same, sorted asc |

---

## Page 3 — Model Explainability

**Purpose:** Show which features drive market value predictions.

### Visuals

| Visual | Type | Fields |
|--------|------|--------|
| Feature Importance | Horizontal bar | `feature` vs `importance`, sorted desc |
| Value by Age (scatter) | Scatter | X=`age`, Y=`market_value_eur`, size=`predicted_value_eur`, color=`position_group` |
| Value by Goals/90 | Scatter | X=`goals_p90`, Y=`predicted_value_eur`, tooltip=`player_name` |
| Avg Predicted Value by Position | Clustered bar | X=`position_group`, Y=Avg `predicted_value_eur` |

---

## Page 4 — League Comparison

**Purpose:** Compare prediction accuracy and player value distributions across PL, La Liga, Serie A.

### Visuals

| Visual | Type | Fields |
|--------|------|--------|
| Avg Market Value by League | Bar chart | `competition` vs Avg `market_value_eur` |
| Avg Predicted Value by League | Bar chart | `competition` vs Avg `predicted_value_eur` |
| % Error Distribution by League | Box plot | `competition` (axis), `abs_pct_error` (value) |
| Top Players per League | Matrix | `competition`, `player_name`, `market_value_eur`, `predicted_value_eur` |

---

## Conditional Formatting Rules

Apply to `pct_error` and `abs_pct_error` columns in all tables:
- Green background: `abs_pct_error` < 10
- Amber background: `abs_pct_error` between 10 and 25  
- Red background: `abs_pct_error` > 25

---

## Scheduled Refresh

Once the Databricks pipeline runs on a schedule (Databricks Jobs → daily trigger at 3 AM):
1. In PowerBI Service: Dataset → Settings → Scheduled Refresh
2. Set to daily, 6 AM (after Databricks job completes)
3. Use Databricks connector with service principal credentials for the gateway
