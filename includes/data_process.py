import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_dataset(df, window_size=14, split_ratio=0.8, scale=True):
    """
    Prepares X (features) and Y (target) for LSTM/Deep Learning models.
    Dynamically includes ALL numeric columns (OHLCV + Indicators) as features.
    
    Args:
        df: DataFrame with computed indicators
        window_size: Number of past days to look at (N)
        split_ratio: % of data to use for training (e.g., 0.8 for 80%)
        scale: Whether to normalize data between 0 and 1 (Default: True)
        
    Returns:
        X_train, y_train, X_test, y_test, scaler (object or None)
    """
    # 1. Define Target (Y)
    # Predict if Next Day's Close > Today's Close
    df = df.copy()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop the last row because it has no 'Next Day' to compare to
    df = df.dropna()

    # 2. Dynamic Feature Selection (Use Head/Columns)
    # We ONLY exclude the Date (metadata) and the Target (the answer key).
    # This means Open, High, Low, Close, Volume, and ALL indicators will be included.
    exclude_cols = ['Date', 'Target']
    
    # "feature_cols" will automatically grab everything else in the dataframe
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Safety Check
    if not feature_cols:
        raise ValueError("No features found! DataFrame might be empty or only contain Date/Target.")

    print(f"\n--- 🔍 Feature Selection ---")
    print(f"Total Features Detected: {len(feature_cols)}")
    print(f"X Features Used: {feature_cols}")  # <--- PRINTING FEATURE NAMES HERE
    
    data = df[feature_cols].values
    targets = df['Target'].values

    # 3. Split Train/Test (Time-based split)
    split_idx = int(len(data) * split_ratio)
    
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    train_targets = targets[:split_idx]
    test_targets = targets[split_idx:]

    # 4. Scale Data (Essential when mixing Price ~100k and RSI ~100)
    if scale:
        # Fit scaler ONLY on training data to avoid data leakage
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_processed = scaler.fit_transform(train_data)
        test_processed = scaler.transform(test_data)
    else:
        scaler = None
        train_processed = train_data
        test_processed = test_data

    # 5. Windowing Function (Sliding Window)
    def create_sliding_window(X_data, y_data, window):
        X, y = [], []
        # We stop a bit earlier to avoid running out of Y values
        for i in range(len(X_data) - window):
            # Input: indices i to i+window (exclusive) -> Length = window
            X.append(X_data[i : i + window])
            
            # Target: The target associated with the LAST day of the window
            # If X ends at index 'i+window-1', we want the target at 'i+window-1'
            # because df['Target'] at row T already predicts T+1.
            y.append(y_data[i + window - 1]) 
            
        return np.array(X), np.array(y)

    X_train, y_train = create_sliding_window(train_processed, train_targets, window_size)
    X_test, y_test = create_sliding_window(test_processed, test_targets, window_size)

    return X_train, y_train, X_test, y_test, scaler