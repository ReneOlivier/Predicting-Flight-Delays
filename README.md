# ✈️ Flight Delay Prediction

A comparative machine learning study predicting whether a US domestic flight will arrive more than 15 minutes late. Three classification models — **Logistic Regression**, **Random Forest**, and **CatBoost** — are trained on the same pipeline and evaluated side by side to identify the strongest performer and understand the trade-offs between them.

---

## 📊 Dataset

**Source:** [US Bureau of Transportation Statistics — On-Time Performance](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD), January 2020

- ~600,000 domestic flight records
- **Target variable:** `ARR_DEL15` — whether a flight arrived 15+ minutes late (binary)
- Class split: ~79% on time, ~21% delayed

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Class balance check to identify dataset imbalance
- Delay rate by carrier — which airlines delay most
- Delay rate by departure hour — identifying peak delay windows
- Top 20 worst-performing routes by delay rate

### 2. Preprocessing
- Dropped a junk trailing column (`Unnamed: 21`)
- Removed ~1% of rows with nulls in key fields (`TAIL_NUM`, `DEP_TIME`, `ARR_TIME`, `ARR_DEL15`)
- `TAIL_NUM` excluded from features due to high cardinality (~4,000 unique values) with minimal predictive signal
- EDA-only columns (`ROUTE`, `DEP_HOUR`) dropped before encoding
- **OneHotEncoding** on categorical features: `OP_UNIQUE_CARRIER`, `OP_CARRIER`, `ORIGIN`, `DEST`, `DEP_TIME_BLK`
- **LabelEncoding** on the target variable
- 80/20 stratified train/test split (`random_state=42`)
- **StandardScaler** with `with_mean=False` (sparse-safe)

### 3. Models

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline — fast, interpretable, lower-bound benchmark |
| Random Forest | 10-tree ensemble with entropy criterion — handles non-linear relationships |
| CatBoost | Gradient boosting with strong categorical handling — minimal tuning required |

### 4. Evaluation
- Accuracy and 10-fold cross-validation mean
- Precision, recall, and F1 score
- Confusion matrices (side by side)
- ROC curves (all three models on a single chart with AUC scores)

---

## 📈 Results

| Model | Accuracy | CV Mean | AUC |
|---|---|---|---|
| Logistic Regression | 93.05% | 93.06% | 0.8965 |
| Random Forest | 92.81% | 92.65% | 0.9215 |
| **CatBoost** | **94.18%** | **94.14%** | **0.9588** |

**CatBoost is the strongest performer across all metrics.** While accuracy scores are close, the AUC gap is significant — CatBoost (0.9588) substantially outperforms both alternatives, meaning it is better at correctly ranking delayed flights across all decision thresholds.

Accuracy alone is a limited metric given the class imbalance — a naive model predicting "on time" for every flight would still achieve ~79% accuracy. Inspecting the confusion matrices reveals a meaningful trade-off: Random Forest produces the fewest false positives (2,068) but misses the most actual delays (6,550 false negatives). CatBoost strikes the better practical balance — 4,758 missed delays against only 2,214 false alarms — making it the most deployable model where missing a genuine delay is typically more costly than a false alarm.

---

## 🛠️ Requirements

```bash
pip install numpy pandas matplotlib scikit-learn catboost
```

Python 3.8+ recommended.

---

## 🚀 Usage

1. Clone the repository
2. Place `jan_2020_ontime.csv` in the same directory as the notebook
3. Open `Flight_Delay_Prediction.ipynb` in Jupyter or VS Code
4. Run all cells from top to bottom

---

## 🔭 Future Improvements

**Modelling**
- Tune CatBoost hyperparameters (learning rate, depth, iterations) using `Optuna` or `GridSearchCV` — the current configuration uses defaults, leaving performance on the table
- Adjust the classification threshold below 0.5 to further reduce false negatives at an acceptable cost to precision, optimising for recall on the positive class
- Apply SMOTE oversampling or `class_weight='balanced'` to explicitly address the class imbalance rather than relying on the model to handle it implicitly

**Features**
- Expand beyond January 2020 to capture seasonal patterns — winter weather, holiday congestion, and summer thunderstorm season all affect delay rates differently
- Engineer route-level historical delay rates as a feature — a flight's route history is likely a strong predictor
- Flag public holidays and school holiday periods as binary features
- Include weather data at origin and destination airports (wind speed, precipitation, visibility)

**Evaluation**
- Report PR-AUC (Precision-Recall AUC) alongside ROC-AUC — more informative on imbalanced datasets
- Add SHAP value analysis to identify which features drive predictions most — adds explainability and portfolio depth

**Deployment**
- Wrap the CatBoost model in a Streamlit app for interactive delay probability predictions
- Serialise the trained model with `joblib` and expose it via a simple FastAPI endpoint

---

## 📄 Data Source

[Bureau of Transportation Statistics — Airline On-Time Statistics](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD)

---

## 📝 License

This project is for educational and portfolio purposes.
