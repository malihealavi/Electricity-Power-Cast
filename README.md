<div align="center">

# ⚡ Electricity Power Cast

### Accurate Electricity Price Forecasting for the German & European Market

*A machine learning pipeline: EDA → feature engineering → imputation → stacked ensemble forecasting*
 
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-f7931e?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
 
<table>
  <tr>
    <td align="center">
      <img src="images/Picture2.png" width="400"><br>
      <sub>Renewable Generation vs Electricity Price</sub>
    </td>
    <td align="center">
      <img src="images/Picture1.png" width="400"><br>
      <sub>Conventional Generation vs Electricity Price</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/Picture5.png" width="400"><br>
      <sub>Total Price vs Actual and Forecast Generation</sub>
    </td>
    <td align="center">
      <img src="images/Picture6.png" width="400"><br>
      <sub>Features vs Electricity Price</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/SHAP.png" width="400"><br>
      <sub>XGBoost SHAP Analysis</sub>
    </td>
    <td align="center">
      <img src="images/IMPORTANCE.png" width="400"><br>
      <sub>XGBoost Feature Importance</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="images/Picture10.png" width="400"><br>
      <sub>Sweden Day ahead price (actual vs predicted)</sub>
    </td>
    <td align="center">
      <img src="images/Picture11.png" width="400"><br>
      <sub>Netherland Day ahead price (actual vs predicted)</sub>
    </td>
  </tr>
</table>

</div>

---

## 📌 Overview

Electricity prices are shaped by many forces at once: supply, demand, the mix of renewable and conventional generation, cross-border trade, and grid-balancing costs. This project uses high-resolution data from **[SMARD](https://www.smard.de/)** (Germany's electricity market platform) to understand these forces and to **forecast day-ahead electricity prices for 16 European bidding zones**.

The project covers the full data science workflow:

1. **Exploratory Data Analysis** of prices, generation, consumption, cross-border flows, and balancing costs
2. **Feature engineering** with new features, external weather and gas-price data, and smart missing-value handling
3. **Model-based imputation** for key features with large gaps
4. **Stacked ensemble forecasting** (Random Forest + XGBoost, combined with Linear / Ridge stackers)
5. **Interpretability** with SHAP, LIME, and feature importance

> 📄 A full written report with all figures and analysis is available in this repository: `ElectricityPowerReport.pdf`.

---

## 🎯 Key Results

| Model | MAE (€/MWh) | RMSE (€/MWh) | R² |
|---|:---:|:---:|:---:|
| Random Forest | 7.45 | 13.26 | 0.927 |
| XGBoost | 8.38 | 14.14 | 0.916 |
| **Stacked Ensemble (Linear Regression)** | **7.00** | **12.51** | **0.936** |
| **Stacked Ensemble (Ridge)** | **7.00** | **12.51** | **0.936** |

**Directional accuracy of the stacked ensemble** (does the price go up or down?):

| Rise | Fall | Stable | Overall |
|:---:|:---:|:---:|:---:|
| ~91.6% | ~92.4–92.6% | ~12% | ~91.0–91.3% |

- The model detects **extreme price movements** (>15% change) with 100% accuracy on the test data.
- Directional accuracy ranges from **86.96% (Sweden)** to **96.67% (DE/LU)** across the 15 evaluated zones.
- The ensemble beats each single model, and the choice of stacker (Linear vs. Ridge) makes almost no difference.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | SMARD (Bundesnetzagentur) |
| **Period** | January 2023 – March 2025 |
| **Files** | 16 CSV files (yearly, monthly, hourly, and quarter-hourly data) |
| **Themes** | Day-ahead prices, generation (actual & forecast), consumption, installed capacity, cross-border physical flows, scheduled commercial exchanges, balancing reserves, TSO costs |
| **Merged data** | 76,216 rows × 130 columns (quarter-hourly) |
| **Final modeling data** | 19,052 rows × 212 columns (hourly) |

**External data added:**
- 🌤️ **Weather** (temperature, wind speed, solar radiation) via the [Meteostat](https://meteostat.net/) API
- ⛽ **Henry Hub natural gas spot price** from the U.S. Energy Information Administration (EIA)

---

## 🔍 Exploratory Data Analysis — Highlights

- **Electricity Price Volatility and Seasonal Trends:** Electricity prices across European countries exhibit significant volatility, with values ranging from €0/MWh to €400/MWh. Seasonal trends are evident, with winter months showing higher prices due to increased demand for heating. Peaks and troughs in prices are influenced by demand surges, supply shortages, and geopolitical events.
- **Impact of Renewable vs. Conventional Energy Sources on Prices:**  Renewable energy sources like wind and solar tend to lower electricity prices when available, while conventional sources such as fossil gas, lignite, and coal are associated with higher prices due to their higher financial and environmental costs. The balance between renewable and conventional sources significantly impacts pricing trends.
- **Electricity Consumption Patterns:**  Consumption follows distinct daily cycles, with lower usage during nighttime hours (0:00–6:00) and peaks in the morning and evening. Seasonal variations show higher consumption during winter months due to heating needs. Residual load (non-renewable energy demand after accounting for renewables) fluctuates more significantly compared to total grid load1.
-	**Cross-Border Energy Flows:**  Germany's energy imports and exports vary significantly by country. Denmark has the highest imports (493 MWh), while Austria leads in exports (282 MWh). Cross-border flows highlight Germany's role as a major hub for energy exchange in Europe.
-	**Generation Capacity Trends:** Installed generation capacity has shown a slight year-on-year increase, reflecting rising energy demands. Wind onshore remains the dominant source of electricity generation in Germany, followed by photovoltaics, fossil gas, and lignite1.
-	**Correlation Between Energy Sources and Prices:** Conventional sources like lignite, coal, and fossil gas show strong positive correlations with electricity prices due to fuel costs and carbon pricing. Renewables generally exhibit negative correlations with price levels, indicating their cost-reducing effect when available.
-	**Transmission System Operator Costs:** Monthly costs incurred by Transmission System Operators (TSOs) for grid stability vary significantly. Network security costs are the highest and exhibit dramatic fluctuations over time, while countertrading activities remain relatively stable


---

## 🛠️ Feature Engineering

**Data cleaning**
- Merged all datasets on `End date`
- Removed duplicate timestamps (4 dates), keeping the most recent record
- Resampled quarter-hourly data to hourly means; back-filled monthly and yearly data

**New features**
- **Forecast error** (actual vs. forecast) for generation and consumption
- **Aggregated generation:** renewable, conventional, and total
- **Net flow** for commercial exchanges and cross-border flows, per country
- **Temporal:** hour, day of week, month
- **Lagged prices:** 24-hour and 168-hour lags, plus 7-day rolling standard deviation
- **Ratios:** `renewable_ratio`, `net_export_ratio`
- **Regional price averages** from neighboring countries (`neighbor_price_avg_*`)

**Feature selection**
- Kept features with |correlation| ≥ 0.5 to prices
- Removed 19 highly inter-correlated features (|corr| > 0.9)

**Missing-value strategy** (127 of 212 columns had gaps)

| Missing share | Columns | Approach |
|---|:---:|---|
| < 5% | 87 | Filled with the median |
| 5% – 80% | 23 | Trained XGBoost imputers; **5** reached good accuracy and were filled, the rest were kept as-is |
| > 80% | 17 | Dropped |

**Imputation model quality (R²)**

| Feature | R² |
|---|:---:|
| Natural gas price | 0.512 |
| FCR procurement price | 0.850 |
| aFRR procurement price (+) | 0.963 |
| mFRR procurement price (+) | 0.834 |
| mFRR procurement price (−) | 0.933 |

---

## 🤖 Modeling Approach

```
        ┌──────────────────┐     ┌──────────────────┐
        │  Random Forest   │     │     XGBoost      │      ← base models
        └────────┬─────────┘     └────────┬─────────┘        (handle missing values)
                 └───────────┬────────────┘
                             ▼
              ┌──────────────────────────────┐
              │ Stacker: Linear Regression / │              ← meta-models
              │            Ridge             │
              └──────────────┬───────────────┘
                             ▼
                  Day-ahead price forecast
                    (per bidding zone)
```

- Features were standardized; **targets were kept in their original scale (€/MWh)**.
- Base models were chosen because they can handle missing values.
- Evaluated with MAE, RMSE, R², directional accuracy (rise / fall / stable), and extreme-movement detection.

### 🔎 What drives the predictions? (SHAP & LIME)

- **Neighbor price average** → strongest positive impact
- **Conventional generation** (especially lignite) and **residual load** → strong positive impact
- **TSO countertrading costs** → negative impact
- **Lagged prices of neighbors** (e.g., Norway, Sweden) → smaller but still significant

---

## ⚠️ Limitations

- **Stable periods are hard to predict** (~12% accuracy). The model is better at detecting trends than flat periods.
- **Sudden sharp spikes and drops** are captured in direction but not always in exact value.
- **Cross-border flows are treated as fixed inputs**, not as responding to prices.
- **The 100% extreme-movement accuracy** should be read with care, since extreme events are rare in the test data.
- **Short history:** only about two years of data, so rare events and long-term trends are limited.

---

## 🚀 Future Work

- Use real **weather forecasts** (not only historical weather) for wind and solar
- Model **cross-border flows dynamically** as price-responsive
- Move to **finer time resolution** for intra-hour use cases
- Add **adaptive / online learning** so the model keeps up with changing grid conditions
- Add **explainability modules** to support regulatory needs (e.g., EU AI Act)

**Potential use cases:** trading support, industrial load-shifting, and grid reserve planning.

---

## 📁 Repository Structure

```
Electricity-Power-Cast/
│
├── src/                                    # All notebooks
│   ├── EDA1.ipynb                          # Exploratory data analysis (part 1)
│   ├── EDA2.ipynb                          # Exploratory data analysis (part 2)
│   ├── EDA_3.ipynb                         # Exploratory data analysis (part 3)
│   ├── Feature_Engineering.ipynb           # Merging, new features, external data, missing values
│   ├── In_depth_analysis.ipynb             # Feature correlations & model-based imputation
│   ├── Price_Prediction_Models.ipynb       # Random Forest & XGBoost base models
│   └── model_aggregated.ipynb              # Stacked ensemble (Linear / Ridge) & evaluation
│
├── images/                                 # Figures used in this README
├── ElectricityPowerReport.pdf              # Full project report
└── README.md
```

---

## ▶️ Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/malihealavi/Electricity-Power-Cast.git
cd Electricity-Power-Cast
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap lime meteostat jupyter
```

**3. Run the notebooks in this order**

| Step | Notebook | Purpose |
|:---:|---|---|
| 1 | `EDA1.ipynb`, `EDA2.ipynb`, `EDA_3.ipynb` | Understand the data |
| 2 | `Feature_Engineering.ipynb` | Build the modeling dataset |
| 3 | `In_depth_analysis.ipynb` | Impute key features |
| 4 | `Price_Prediction_Models.ipynb` | Train and evaluate base models |
| 5 | `model_aggregated.ipynb` | Train and evaluate the stacked ensemble |

> The raw SMARD data can be downloaded from [smard.de](https://www.smard.de/en/downloadcenter/download-market-data).

---

## 🧰 Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `SHAP` · `LIME` · `Matplotlib` · `Seaborn` · `Meteostat` · `Jupyter`

---

<div align="center">

⭐ If you find this project useful, consider giving it a star!

</div>
