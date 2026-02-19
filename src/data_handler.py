import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start, end):
    """Fetches historical stock data from Yahoo Finance."""
    print(f"Fetching {ticker} stock data...")
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        raise ValueError(f"No data fetched for {ticker}. Check ticker or connection.")
    print(f"Data fetched successfully. Shape: {data.shape}")
    return data

def clean_data(data):
    """Removes NaN values and logs changes."""
    original_shape = data.shape
    data_cleaned = data.dropna()
    print(f"Data cleaned. Shape changed from {original_shape} to {data_cleaned.shape}")
    return data_cleaned