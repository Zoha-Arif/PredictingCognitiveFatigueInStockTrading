import yfinance as yf
import pandas as pd
import random
from tqdm import tqdm

# FakeTrader models a trader whose fatigue increases over time
class FakeTrader:
    def __init__(self):
        self.reaction_time = 1.0  # seconds
        self.fatigue_score = 0.0  # builds up as session progresses
    
    def decide(self, row, i):
        self.reaction_time += 0.02
        if i % 20 == 0:
            self.fatigue_score += 0.1
    
        try:
            price_change = float(row['Close']) - float(row['Open'])
        except:
            price_change = 0.0
    
        noise = random.uniform(-0.3, 0.3)
    
        if price_change + noise > 0.2:
            action = "BUY"
        elif price_change + noise < -0.2:
            action = "SELL"
        else:
            action = "HOLD"
    
        is_fatigued = int(self.reaction_time > 2.5 or self.fatigue_score > 1.0)
    
        return {
            "decision": action,
            "reaction_time": round(self.reaction_time, 2),
            "fatigue": is_fatigued
        }

# Load 1-day of 1-minute AAPL stock data
def fetch_stock_data(symbol="AAPL"):
    print(f"Downloading stock data for {symbol}...")
    df = yf.download(symbol, period="1d", interval="1m")
    return df.reset_index()[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

# Simulate the trading session
def simulate_trading():
    data = fetch_stock_data("AAPL")
    trader = FakeTrader()
    logs = []

    for i, row in tqdm(data.iterrows(), total=len(data)):
        result = trader.decide(row, i)
        logs.append({
            "timestamp": row["Datetime"],
            "open": row["Open"],
            "close": row["Close"],
            "decision": result["decision"],
            "reaction_time": result["reaction_time"],
            "fatigue": result["fatigue"]
        })

    df = pd.DataFrame(logs)
    df.to_csv("data/simulated_trading_behavior.csv", index=False)
    print("Simulation complete...output saved to data/simulated_trading_behavior.csv")

if __name__ == "__main__":

    simulate_trading()
