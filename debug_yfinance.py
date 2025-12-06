
import yfinance as yf
import pandas as pd

def test_yfinance(ticker):
    print(f"Testing ticker: {ticker}")
    print(f"Yfinance version: {yf.__version__}")
    try:
        print("--- Testing Ticker.history() ---")
        stock = yf.Ticker(ticker)
        history = stock.history(period="1mo")
        print(f"History fetched via Ticker. Empty? {history.empty}")
        
        print("\n--- Testing yf.download() ---")
        data = yf.download(ticker, period="1mo", progress=False)
        print(f"Data fetched via download. Empty? {data.empty}")
        if not data.empty:
            print(f"First row: {data.iloc[0]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_yfinance("AAPL")
