import pandas as pd
import numpy as np

def calculate_rsi(df, period=14):
    """
    Relative Strength Index (RSI)
    Measures the speed and change of price movements.
    Range: 0-100
    """
    # Calculate price changes
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Calculate Smoothed Averages (Wilder's Smoothing)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Moving Average Convergence Divergence (MACD)
    Returns two series: MACD Line and Signal Line.
    """
    # Calculate Fast and Slow EMAs
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    # MACD Line
    macd_line = ema_fast - ema_slow
    
    # Signal Line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    return macd_line, signal_line

def calculate_williams_r(df, period=14):
    """
    Williams %R
    Momentum indicator inversely related to Stochastic Oscillator.
    Range: -100 to 0
    """
    # Rolling Highest High and Lowest Low
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    
    # Formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
    wr = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    
    return wr

def calculate_bollinger_b(df, period=20, std_dev=2):
    """
    Bollinger Band %B
    Quantifies a security's price relative to the upper and lower Bollinger Band.
    > 1.0: Above Upper Band
    < 0.0: Below Lower Band
    """
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    # %B Formula: (Price - Lower) / (Upper - Lower)
    b_percent = (df['Close'] - lower_band) / (upper_band - lower_band)
    
    return b_percent

def calculate_natr(df, period=14):
    """
    Normalized Average True Range (NATR)
    Volatility indicator normalized by price (percentage terms).
    Useful for comparing volatility across different price levels.
    """
    # True Range Calculation
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    
    # True Range is the max of the three
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Average True Range
    atr = tr.rolling(window=period).mean()
    
    # Normalize: (ATR / Close) * 100
    natr = (atr / df['Close']) * 100
    
    return natr

def calculate_volume_roc(df, period=1):
    """
    Volume Rate of Change (V-ROC)
    Measures the percentage change in volume.
    """
    v_roc = df['Volume'].pct_change(periods=period) * 100
    return v_roc



def calculate_vol_ratio(df, period=14):
    """
    Volatility Ratio (Vol_Ratio)
    Compares Today's True Range to the Historical Average True Range.
    
    Formula: Today's TR / Previous N-day ATR
    - 1.0 = Volatility is normal
    - > 2.0 = Volatility Explosion (Breakout/Crash)
    - < 0.5 = Volatility Compression (Calm before storm)
    """
    # 1. True Range Calculation (Same as NATR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    
    # True Range is the max of the three
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # 2. Average True Range (ATR)
    # We use the previous 'period' days to establish the baseline
    atr = tr.rolling(window=period).mean()
    
    # 3. Calculate Ratio
    # We divide Today's TR by the PREVIOUS day's ATR baseline.
    # We shift ATR by 1 to compare "Today's Reality" vs "Yesterday's Expectation"
    vol_ratio = tr / atr.shift(1)
    
    return vol_ratio