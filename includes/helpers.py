import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pickle
import json
import os


def save_artifacts(model, scaler, threshold, filename_prefix="btc_lstm"):
    """
    Universally saves models (Keras, XGBoost, Sklearn), scalers, and config 
    to the 'saved_artifacts' folder.
    """
    # 1. Ensure the folder exists
    folder = "saved_artifacts"
    os.makedirs(folder, exist_ok=True)
    
    print(f"Saving artifacts to '{folder}/' with prefix '{filename_prefix}'...")

    # 2. Save the Model (Auto-Detect Type)
    # -------------------------------------------------------
    if hasattr(model, 'save'):
        # CASE A: Keras or XGBoostAdapter (Has a .save method)
        # Note: The adapter internally handles the swap to .json if needed
        model_path = f"{folder}/{filename_prefix}_model.keras"
        model.save(model_path)
        print(f"   -> Model saved to {model_path} (or .json)")
        
    else:
        # CASE B: Random Forest / Standard Sklearn (No .save method)
        # We must use Pickle
        model_path = f"{folder}/{filename_prefix}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"   -> Model saved to {model_path} (Pickle)")

    # 3. Save the Scaler (Always Pickle)
    # -------------------------------------------------------
    scaler_path = f"{folder}/{filename_prefix}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"   -> Scaler saved to {scaler_path}")

    # 4. Save the Threshold (Always JSON)
    # -------------------------------------------------------
    config_path = f"{folder}/{filename_prefix}_config.json"
    config = {"optimal_threshold": float(threshold)}
    with open(config_path, "w") as f:
        json.dump(config, f)
    print(f"   -> Config saved to {config_path}")

    print("✅ All artifacts saved successfully!")


def backtest_strategy(model, X_test, df_test, threshold=0.4848):
    """
    Simulates trading based on model predictions.
    
    Args:
        model: Trained LSTM
        X_test: Input features
        df_test: The original dataframe corresponding to X_test (needs 'Close' price)
        threshold: The dynamic threshold we calculated (0.4437)
    """
    # 1. Get Predictions
    probs = model.predict(X_test, verbose=0).flatten()
    signals = (probs > threshold).astype(int)
    
    # 2. Setup Simulation
    initial_capital = 1000
    cash = initial_capital
    position = 0 # 0 = No Bitcoin, 1 = Holding Bitcoin
    equity_curve = []
    
    # Align prices (We need the Close price of the day we act on)
    # Note: X_test corresponds to the END of the window. 
    # We trade at the Close of that day (or Open of next). 
    # For simplicity, we calculate daily returns.
    
    prices = df_test['Close'].values
    
    for i in range(len(signals) - 1):
        # Current Price
        price_today = prices[i]
        price_tomorrow = prices[i+1]
        
        # Strategy Logic
        if signals[i] == 1 and position == 0:
            # Buy Signal + We have cash -> BUY
            position = cash / price_today
            cash = 0
            
        elif signals[i] == 0 and position > 0:
            # Sell Signal + We have BTC -> SELL
            cash = position * price_today
            position = 0
            
        # Calculate Daily Value
        current_equity = cash + (position * price_today)
        equity_curve.append(current_equity)
    
    # 3. Benchmark (Buy & Hold)
    # If we just bought $1000 of BTC on Day 1 and held it
    btc_return = (prices[-1] - prices[0]) / prices[0]
    buy_hold_final = initial_capital * (1 + btc_return)
    
    model_final = equity_curve[-1]
    
    # 4. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='AI Model Strategy', color='blue')
    
    # Plot Benchmark (Simple line from start to end)
    plt.plot([0, len(equity_curve)], [initial_capital, buy_hold_final], 
             label='Buy & Hold (Benchmark)', color='gray', linestyle='--')
    
    plt.title(f'Backtest: AI vs Buy & Hold (Start: ${initial_capital})')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Days')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"--- Final Results ---")
    print(f"Initial Investment: ${initial_capital}")
    print(f"Buy & Hold Final:   ${buy_hold_final:.2f}")
    print(f"AI Model Final:     ${model_final:.2f}")




def diagnose_model_output(model, X_test):
    # 1. Get raw probability scores (0.0 to 1.0)
    raw_preds = model.predict(X_test, verbose=0)
    
    # 2. Statistics
    print(f"--- Prediction Statistics ---")
    print(f"Max Prob: {raw_preds.max():.4f}")
    print(f"Min Prob: {raw_preds.min():.4f}")
    print(f"Mean Prob: {raw_preds.mean():.4f}")
    
    # 3. Plot Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(raw_preds, bins=50, color='purple', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.title('Distribution of Model Predictions (The "Confidence" Check)')
    plt.xlabel('Probability Score (0 = Down, 1 = Up)')
    plt.ylabel('Count')
    plt.legend()
    plt.show()



def get_crypto_data(ticker, start_date, end_date, interval="1d"):
    """
    Fetches OHLCV data for a specific date range.
    
    Args:
        ticker (str): The symbol to fetch (e.g., 'BTC-USD')
        start_date (str): Format 'YYYY-MM-DD'
        end_date (str): Format 'YYYY-MM-DD' (Note: yfinance excludes the end date)
        interval (str): The data frequency (e.g., '1d')
        
    Returns:
        pd.DataFrame: A dataframe with Open, High, Low, Close, Volume
    """
    print(f"Fetching {ticker} data from {start_date} to {end_date}...")
    
    # Download data with specific start and end dates
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)
    
    # Handle multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Select only the columns we need
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = data[required_cols].copy()
    
    # Handle missing values
    df = df.ffill()
    
    return df





def plot_crypto(df, title="BTC-USD Price & Volume"):
    # Create subplots: Top for Price (3 parts), Bottom for Volume (1 part)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Data Preparation for Colors ---
    # Define "Up" days (Close > Open) and "Down" days
    up = df[df['Close'] >= df['Open']]
    down = df[df['Close'] < df['Open']]
    
    # Colors
    col_up = 'green'
    col_down = 'red'
    width = .6  # Width of the candle bars
    width2 = .05 # Width of the wick lines

    # --- Plotting Candlesticks (Top Chart) ---
    # Plot "Up" candles
    ax1.bar(up.index, up['Close'] - up['Open'], width, bottom=up['Open'], color=col_up)
    ax1.bar(up.index, up['High'] - up['Close'], width2, bottom=up['Close'], color=col_up)
    ax1.bar(up.index, up['Low'] - up['Open'], width2, bottom=up['Open'], color=col_up)
    
    # Plot "Down" candles
    ax1.bar(down.index, down['Close'] - down['Open'], width, bottom=down['Open'], color=col_down)
    ax1.bar(down.index, down['High'] - down['Open'], width2, bottom=down['Open'], color=col_down)
    ax1.bar(down.index, down['Low'] - down['Close'], width2, bottom=down['Close'], color=col_down)

    # --- Plotting Volume (Bottom Chart) ---
    ax2.bar(up.index, up['Volume'], color=col_up, alpha=0.5)
    ax2.bar(down.index, down['Volume'], color=col_down, alpha=0.5)

    # --- Formatting ---
    ax1.set_title(title)
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)

    # Format Date Axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()




def check_indicators(df, n_rows=200):
    """
    Plots 'Close' price plus ANY other columns found in the dataframe 
    that are not standard OHLCV data.
    """
    # 1. Slice data
    if n_rows and n_rows < len(df):
        plot_df = df.tail(n_rows)
    else:
        plot_df = df

    # 2. Identify what to plot by looking at the dataframe headers
    # Standard columns to exclude from indicator panels
    # Note: We include 'Price' in exclude if it's just a duplicate of Close, 
    # but we plot 'Close' explicitly below.
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Target', 'Adj Close']
    
    # "The List": Every column in your DF that isn't a standard price column
    indicator_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 3. Setup Grid
    # 1 row for Price + N rows for indicators
    total_plots = 1 + len(indicator_cols)
    
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 3 * total_plots), sharex=True)
    
    # Handle single-plot case (if no indicators found)
    if total_plots == 1:
        axes = [axes]

    # --- PLOT 1: PRICE ---
    # We always start with the asset price for context
    axes[0].plot(plot_df.index, plot_df['Close'], label='Close Price', color='black')
    axes[0].set_title(f'Price Action (Last {len(plot_df)} bars)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')

    # --- PLOT 2+: EVERYTHING ELSE FOUND IN DF ---
    for i, col_name in enumerate(indicator_cols):
        ax = axes[i + 1]
        
        # Simple Logic: 
        # If it looks like a histogram (Volume/ROC), use bars.
        # Otherwise, use a standard line plot.
        if "Volume" in col_name or "ROC" in col_name:
            ax.bar(plot_df.index, plot_df[col_name], color='gray', alpha=0.8, label=col_name)
        else:
            ax.plot(plot_df.index, plot_df[col_name], color='tab:blue', label=col_name)
            
        # Add a zero line if the data oscillates around 0 (simple heuristic)
        # We check if min is negative and max is positive
        if plot_df[col_name].min() < 0 and plot_df[col_name].max() > 0:
            ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)

        ax.set_title(col_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    
def plot_training_loss(history):
    """
    Plots the loss curves from the Keras history object.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    
    # Plot validation loss
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    
    plt.title('Model Loss: Train vs Validation')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



def plot_confusion_matrix_percent(model, X_test, y_test):
    """
    Plots the Confusion Matrix with Percentages.
    """
    # 1. Get Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = (y_pred_probs > 0.5).astype(int)
    
    # 2. Calculate Matrix (Normalized)
    # normalize='all' converts counts to % of total dataset
    # normalize='true' would show Recall (Row accuracy)
    # normalize='pred' would show Precision (Column accuracy)
    cm_percent = confusion_matrix(y_test, y_pred_classes, normalize='all')
    
    # Calculate raw counts just for the printed report
    cm_raw = confusion_matrix(y_test, y_pred_classes)
    
    # 3. Plot
    plt.figure(figsize=(8, 6))
    
    # values_format='.2%' tells it to display 0.54 as "54.00%"
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, 
                                  display_labels=['Down', 'Up'])
    
    disp.plot(cmap=plt.cm.Blues, values_format='.2%')
    
    plt.title('Confusion Matrix (Percentage of Total)')
    plt.show()

# 4. Detailed Printout (Raw Counts + Percentages)
    tn, fp, fn, tp = cm_raw.ravel()
    total = cm_raw.sum()

    print(f"\n--- Detailed Report (Total Days: {total}) ---")
    print(f"True Positives  (Up    -> Up)   : {tp} ({tp/total:.2%})")
    print(f"True Negatives  (Down  -> Down) : {tn} ({tn/total:.2%})")
    print(f"False Positives (Down  -> Up)   : {fp} ({fp/total:.2%})  <-- Trap")
    print(f"False Negatives (Up    -> Down) : {fn} ({fn/total:.2%})  <-- Missed Opp")
