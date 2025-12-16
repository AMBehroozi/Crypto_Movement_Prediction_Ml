# 🚀 Bitcoin Price Movement Prediction with Machine Learning

A comprehensive machine learning system for predicting Bitcoin price movements using multiple algorithms and technical indicators. This project demonstrates end-to-end ML pipeline development, from data acquisition to model deployment.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-red.svg)](https://xgboost.readthedocs.io/)

---

If the ML model can predict whether Bitcoin's price will go up or down tomorrow with reasonable accuracy, it could potentially inform trading decisions and help understand market patterns. This project explores that possibility by building a complete machine learning pipeline that analyzes historical price data and technical indicators to forecast next-day price movements.

The approach here is straightforward: we treat price prediction as a binary classification problem. Instead of trying to predict exact prices (which is extremely difficult), we simply ask "will tomorrow's closing price be higher or lower than today's?" This simpler question allows us to use a variety of machine learning algorithms and compare their performance. The system fetches real Bitcoin data, calculates commonly used technical indicators like RSI and MACD, and trains multiple models to find patterns in the data.

What makes this project interesting is the comprehensive comparison of different ML approaches. From deep learning (LSTM) to traditional algorithms (Random Forest, SVM), each model brings its own strengths. The backtesting framework lets you see how each strategy would have performed historically, and the prediction system is ready to use with live data. It's a complete example of taking a machine learning project from data collection all the way to deployment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Indicators](#technical-indicators)
- [Machine Learning Models](#machine-learning-models)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [License](#license)

## 🎯 Overview

This project implements a binary classification system to predict whether Bitcoin's price will move **UP** or **DOWN** the next day. It leverages:

- **5 different ML algorithms** (LSTM, XGBoost, Random Forest, SVM, KNN)
- **11 technical indicators** for feature engineering (6 technical + 4 macroeconomic + 1 volume ratio)
- **Optimized decision thresholds** for each model
- **Comprehensive backtesting framework** with performance comparison
- **Production-ready prediction system** with confidence scoring
- **Model consensus system** for multi-model prediction comparison

## ✨ Features

- 🔄 **Automated data fetching** from Yahoo Finance API
- 📊 **Technical indicator calculation** (RSI, MACD, Williams %R, Bollinger %B, NATR, Volume ROC, Volume Ratio)
- 🌍 **Macroeconomic features** (Dollar Index, S&P 500, Gold, 10-Year Treasury)
- 🤖 **Multiple ML models** with unified training pipeline
- 🎯 **Threshold optimization** for maximum accuracy
- 📈 **Backtesting engine** with Buy & Hold comparison
- 🔍 **Rolling window validation** for robustness testing
- 💾 **Smart artifact management** (auto-save/load models, scalers, configs)
- 🎨 **Rich visualizations** (confusion matrices, training curves, equity curves)
- 🔮 **Live prediction system** with confidence scoring
- 🏆 **Model consensus table** for comparing all 5 models side-by-side

## 📁 Project Structure

```
market_ML_prediction/
│
├── start.ipynb                 # Main training notebook
├── predictor.ipynb             # Production inference system
│
├── includes/
│   ├── helpers.py              # Utility functions (plotting, backtesting, saving)
│   ├── technical_indicators.py # Technical indicator calculations
│   ├── data_process.py         # Dataset creation with sliding windows
│   └── ml_models.py            # Model builders and adapters
│
└── saved_artifacts/            # Trained models and configurations
    ├── btc_lstm_model.keras
    ├── btc_lstm_scaler.pkl
    ├── btc_lstm_config.json
    ├── btc_xgboost_model.json
    ├── btc_rf_model.pkl
    ├── btc_svm_model.pkl
    ├── btc_knn_model.pkl
    └── ... (scalers and configs for each model)
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/market_ML_prediction.git
   cd market_ML_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn xgboost yfinance requests seaborn
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

## 🚀 Usage

### Training Models

Open `start.ipynb` and run the cells sequentially to:

1. **Fetch historical BTC data**
   ```python
   from includes.helpers import get_crypto_data
   btc_data = get_crypto_data("BTC-USD", start_date="2001-01-01", end_date="2025-12-14")
   ```

2. **Calculate technical indicators**
   ```python
   from includes.technical_indicators import *
   btc_data['RSI'] = calculate_rsi(btc_data)
   btc_data['MACD_Line'], btc_data['Signal_Line'] = calculate_macd(btc_data)
   # ... (other indicators)
   ```

3. **Create dataset with sliding windows**
   ```python
   from includes.data_process import create_dataset
   X_train, y_train, X_test, y_test, scaler = create_dataset(
       btc_data, window_size=10, scale=True, split_ratio=0.8
   )
   ```

4. **Train a model** (example: LSTM)
   ```python
   from includes.ml_models import build_lstm_model, train_model
   
   model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
   history, trained_model = train_model(
       model, X_train, y_train, X_test, y_test, 
       epochs=50, batch_size=32
   )
   ```

5. **Optimize threshold and save**
   ```python
   from includes.ml_models import find_optimal_threshold
   from includes.helpers import save_artifacts
   
   optimal_thresh = find_optimal_threshold(trained_model, X_test, y_test)
   save_artifacts(trained_model, scaler, optimal_thresh, "btc_lstm")
   ```

### Making Predictions

Open `predictor.ipynb` to use the trained models:

#### Single Model Prediction
```python
# Initialize predictor with desired model
bot = CryptoPredictor(prefix="btc_lstm")  # or btc_xgboost, btc_rf, btc_svm, btc_knn

# Load artifacts
bot.load_artifacts()

# Predict next day movement
bot.predict_next_day(target_date="2025-12-15")
```

**Output Example:**
```
--- PREDICTION REPORT ---
Data Date:       2025-12-15
Input Price:     $86,206.04 (Reference Level)
---------------------------
Probability (Up): 40.66%
Threshold Used:   48.00%
Margin:           7.34% pts
---------------------------
Prediction:       DOWN 🔴
Confidence:       Medium
```

#### Model Consensus (All 5 Models)
```python
import pandas as pd

model_prefixes = ["btc_lstm", "btc_xgboost", "btc_rf", "btc_svm", "btc_knn"]
target_date = "2025-12-13"

results_list = []
for prefix in model_prefixes:
    bot = CryptoPredictor(prefix)
    bot.load_artifacts()
    stats = bot.predict_next_day(target_date)
    results_list.append(stats)

df = pd.DataFrame(results_list)
print(df.to_string(index=False))
```

**Consensus Output Example:**
```
--- 🏆 MODEL CONSENSUS TABLE ---
  Model Probability Threshold  Trend Confidence
   LSTM      63.38%      0.38   UP 🟢     26.75%
XGBOOST      50.10%      0.51 DOWN 🔴      0.19%
     RF      51.11%      0.48   UP 🟢      2.22%
    SVM      51.70%      0.52 DOWN 🔴      3.41%
    KNN      50.08%      0.45   UP 🟢      0.15%
```

This consensus view helps you see agreement/disagreement across different model architectures.

## 📊 Features & Indicators

The system uses **17 features per time step**, combining price data, technical indicators, and macroeconomic factors:

### Price Features (5)
| Feature | Description |
|---------|-------------|
| **Open** | Opening price |
| **High** | Highest price |
| **Low** | Lowest price |
| **Close** | Closing price |
| **Volume** | Trading volume |

### Technical Indicators (7)
| Indicator | Description | Range |
|-----------|-------------|-------|
| **RSI** | Relative Strength Index | 0-100 |
| **MACD Line** | Moving Average Convergence Divergence | Variable |
| **Signal Line** | MACD Signal Line | Variable |
| **Williams %R** | Momentum indicator | -100 to 0 |
| **Bollinger %B** | Position relative to Bollinger Bands | 0-1 (typically) |
| **NATR** | Normalized Average True Range | Percentage |
| **Volume ROC** | Volume Rate of Change | Percentage |
| **Volume Ratio** | Current volume vs 14-day average | Ratio |

### Macroeconomic Indicators (4)
| Indicator | Description | Ticker |
|-----------|-------------|--------|
| **DXI** | US Dollar Index | DX-Y.NYB |
| **SP500** | S&P 500 Index | ^GSPC |
| **GOLD** | Gold Futures | GC=F |
| **Ten_Y** | 10-Year Treasury Yield | ^TNX |

These macroeconomic features help capture broader market sentiment and correlations that may influence Bitcoin price movements.

## 🤖 Machine Learning Models

### 1. LSTM (Long Short-Term Memory)
- **Type:** Deep Learning / Recurrent Neural Network
- **Architecture:** 50 LSTM units + Dropout(0.2) + Dense(1, sigmoid)
- **Strengths:** Captures sequential patterns and temporal dependencies

### 2. XGBoost (Extreme Gradient Boosting)
- **Type:** Ensemble / Gradient Boosting
- **Parameters:** 100 estimators, max_depth=6, learning_rate=0.05
- **Strengths:** Handles non-linear relationships, feature importance

### 3. Random Forest
- **Type:** Ensemble / Decision Trees
- **Parameters:** 100 estimators, max_depth=10
- **Strengths:** Robust to overfitting, handles noise well

### 4. SVM (Support Vector Machine)
- **Type:** Kernel-based Classification
- **Kernel:** RBF (Radial Basis Function)
- **Strengths:** Effective in high-dimensional spaces

### 5. KNN (K-Nearest Neighbors)
- **Type:** Instance-based Learning
- **Parameters:** 50 neighbors, distance-weighted
- **Strengths:** Pattern matching, no training phase

## 🔬 Methodology

### Data Preparation
1. **Sliding Window Approach:** Uses N previous days (default: 10) to predict day N+1
2. **Train/Test Split:** 80/20 temporal split (no shuffling to prevent lookahead bias)
3. **Feature Scaling:** MinMaxScaler normalization (0-1 range)
4. **Target Definition:** Binary (1 = price goes up, 0 = price goes down)

### Model Training
1. **Class Weight Balancing:** Handles imbalanced bull/bear market data
2. **Validation:** Uses held-out test set for evaluation
3. **Threshold Optimization:** Finds optimal decision boundary (not fixed at 0.5)
4. **Early Stopping:** Prevents overfitting (XGBoost)

### Evaluation
1. **Confusion Matrix Analysis:** True/False Positives and Negatives
2. **Backtesting:** Simulates trading strategy vs Buy & Hold
3. **Rolling Window Validation:** Tests robustness across random time periods
4. **Comparative Analysis:** Benchmarks all models side-by-side

### Adapter Pattern
All sklearn-based models use a **Keras-style adapter** to ensure:
- Unified `.fit()`, `.predict()`, `.evaluate()`, `.save()` interface
- Automatic 3D→2D flattening for non-sequential models
- Compatible with the same training pipeline as LSTM

## 🛠️ Technologies Used

### Core Libraries
- **TensorFlow/Keras** - Deep learning framework (LSTM)
- **XGBoost** - Gradient boosting library
- **scikit-learn** - ML algorithms (RF, SVM, KNN) and preprocessing
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation

### Data & Visualization
- **yfinance** - Yahoo Finance API wrapper
- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical visualizations

### Utilities
- **Pickle** - Model serialization
- **JSON** - Configuration storage
- **Requests** - HTTP requests for live data

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/market_ML_prediction/issues).

## ⚠️ Disclaimer

This project is for **educational purposes only**. The models are not hyperparameter-tuned or optimized for production use—they serve as demonstrations of different ML approaches and pipeline architecture. Cryptocurrency trading involves substantial risk of loss. Do not use this system for actual trading without thorough testing, proper model optimization, and understanding of the risks involved.

---

**Made with ❤️ for the ML and Crypto communities**
