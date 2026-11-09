# ⚡ Electricity Power Cast

Forecasting day-ahead electricity prices for 15 European markets using German SMARD market data, feature engineering, and a stacked ensemble model.

## Overview

Electricity prices change fast. They depend on how much power is made, how much is used, and how much flows across borders. This project uses open data from [SMARD](https://www.smard.de) (Germany's electricity market platform) to find what drives prices and to predict them.

- **Data:** 16 CSV files, Jan 2023 – Mar 2025 (prices, generation, consumption, cross-border flows, balancing reserves, TSO costs)
- **Extra data:** weather (Meteostat API) and Henry Hub natural gas prices (US EIA)
- **Final dataset:** 19,052 hourly rows × 212 features

## Workflow

1. **EDA** – price patterns, seasonality, generation mix, consumption, cross-border flows
2. **Feature engineering** – merge files, remove duplicates, resample to hourly, then add:
   - forecast errors, renewable / conventional / total generation
   - net import-export flows, renewable ratio
   - time features, 24h and 168h price lags, 7-day rolling std
   - average neighbor-country prices
3. **Missing values** – median fill (<5% missing), XGBoost imputation for 5 features (5–80% missing), drop columns with >80% missing
4. **Modeling** – Random Forest and XGBoost as base models, Linear Regression / Ridge as stackers

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 7.45 | 13.26 | 0.927 |
| XGBoost | 8.38 | 14.14 | 0.916 |
| **Stacked (RF + XGBoost → Linear Regression)** | **7.00** | **12.51** | **0.936** |

- Direction accuracy: **~91.6%** for price rises and **~92.4%** for price falls
- Weak spot: stable periods (~12% accuracy) and sudden price spikes

## Key Findings

- Prices are higher in winter and very volatile at the hourly level.
- Conventional sources (lignite, hard coal, gas) go up with prices. Renewables go down with prices.
- Neighbor-country prices and conventional generation are the strongest price drivers (confirmed with SHAP and LIME).
- Prices in most countries are strongly linked (many pairs above 0.9 correlation).

## Repository Structure

| File | Description |
|---|---|
| `EDA1.ipynb`, `EDA2.ipynb`, `EDA_3.ipynb` | Exploratory data analysis |
| `Feature_Engineering.ipynb` | Merging, cleaning, new features |
| `In_depth_analysis.ipynb` | Correlation study and imputation models |
| `Price_Prediction_Models.ipynb` | Random Forest and XGBoost models |
| `model_aggregated.ipynb` | Stacking ensemble and evaluation |
| `dates.csv` | Date index |
| `normalized_features.csv` | Scaled feature set |
| `transformed_imputed_5_features.csv` | Features with imputed values |

## Getting Started

```bash
git clone https://github.com/malihealavi/Electricity-Power-Cast.git
cd Electricity-Power-Cast
pip install pandas numpy scikit-learn xgboost shap lime meteostat matplotlib seaborn jupyter
jupyter notebook
```

Run the notebooks in this order: EDA → Feature Engineering → In-depth analysis → Price Prediction Models → model_aggregated.

> The raw SMARD files are not included. Download them from [smard.de](https://www.smard.de) (Jan 2023 – Mar 2025).

## Limitations and Next Steps

- Predicts well on average, but misses extreme spikes
- Cross-border flows are treated as fixed inputs, not price-responsive
- Planned: weather forecasts as inputs, finer time resolution, models that update themselves over time

## Author

**Malihe Alavi** · Full report: `ElectricityPowerReport.pdf`
