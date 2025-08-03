import torch
import pandas as pd
import numpy as np
from train_model import FatigueLSTM  # import the model class

# Load most recent session
SESSION_CSV = "data/simulated_trading_behavior.csv"
df = pd.read_csv(SESSION_CSV)

# Encode decisions
decision_map = {"HOLD": 0, "BUY": 1, "SELL": 2}
df["decision_encoded"] = df["decision"].map(decision_map)

# Load model
model = FatigueLSTM(input_size=2, hidden_size=64, num_layers=1)
model.load_state_dict(torch.load("fatigue_model.pt"))  # load weights
model.eval()

# Real-time fatigue prediction from rolling window
SEQUENCE_LENGTH = 10
features = ["reaction_time", "decision_encoded"]

print("\n🧠 Real-Time Fatigue Prediction\n")

for i in range(SEQUENCE_LENGTH, len(df)):
    window = df.iloc[i - SEQUENCE_LENGTH : i][features].values
    input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # shape (1, seq_len, features)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    fatigue = "⚠️ FATIGUED" if prediction > 0.5 else "✅ Alert"
    print(f"Step {i:3d} | Fatigue Score: {prediction:.2f} → {fatigue}")