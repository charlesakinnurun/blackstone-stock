import pandas as pd

def create_target_and_features(data):
    """Generates technical indicators and the prediction target."""
    df = data.copy()
    
    # Target: Will the price go up tomorrow?
    df["Price_Change"] = df["Close"].shift(-1) - df["Close"]
    df['Target'] = (df["Price_Change"] > 0).astype(int)

    # Technical Indicators
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Volume_SMA_5"] = df["Volume"].rolling(window=5).mean()
    df["Daily_Change"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Lag_5_Change"] = df["Close"].diff(5)
    
    return df

def get_feature_lists():
    """Returns the list of column names used as features."""
    return ["SMA_5", "SMA_20", "Volume_SMA_5", "Daily_Change", 
            "Lag_5_Change", "Open", "High", "Low", "Close", "Volume"]