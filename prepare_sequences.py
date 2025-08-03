import pandas as pd
import numpy as np

# Settings
SEQUENCE_LENGTH = 10
DATA_PATH = "data/simulated_trading_behavior.csv"
SAVE_X = "data/X.npy"
SAVE_Y = "data/y.npy"

# Load CSV
df = pd.read_csv(DATA_PATH)

# Optionally encode decision as numeric
decision_map = {"HOLD": 0, "BUY": 1, "SELL": 2}
df["decision_encoded"] = df["decision"].map(decision_map)

# Features to use
features = ["reaction_time", "decision_encoded"]
X, y = [], []

# Build sequences
for i in range(len(df) - SEQUENCE_LENGTH):
    window = df.iloc[i : i + SEQUENCE_LENGTH]
    sequence = window[features].values
    label = window["fatigue"].iloc[-1]  # label = fatigue on last timestep
    X.append(sequence)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Save
np.save(SAVE_X, X)
np.save(SAVE_Y, y)

print(f"✅ Saved {X.shape[0]} sequences of shape {X.shape[1:]} to:")
print(f"  → {SAVE_X}")
print(f"  → {SAVE_Y}")