# 🧠 Predicting Cognitive Fatigue in Stock Trading

This project uses a neural network model to predict cognitive fatigue in simulated stock trading behavior based on decision-making patterns and reaction times. It combines time-series processing, deep learning, and trading simulation to explore how fatigue may manifest in behavioral data.

The goal was to build a real-time fatigue detection system that analyzes:
- Reaction time consistency
- Trading decisions over time (BUY, SELL, HOLD)
- Sequence patterns leading to mental fatigue

---

## 🧠 Features

- **FatigueLSTM model**: Custom LSTM-based neural network built using PyTorch  
- **Real-time fatigue prediction** using rolling windows  
- **Simulated trader environment** for generating or testing behavioral data  
- **Behavioral encoding** of decisions for model compatibility  
- **Configurable sequence length and model hyperparameters**  

---

## 📁 Repository Structure

| File                   | Description                                               |
|------------------------|-----------------------------------------------------------|
| `fatigue_model.pt`     | Pretrained PyTorch model weights                          |
| `predict_live.py`      | Real-time prediction using the trained model              |
| `prepare_sequences.py` | Sequence preprocessing for time-series input              |
| `train_model.py`       | Training script for LSTM using session data               |
| `trader_simulator.py`  | Simulates trading sessions with reaction times and choices|
| `requirements.txt`     | Python dependencies                                       |

---

## 🛠 Tech Stack

- **Python**
- **PyTorch**
- **pandas**, **NumPy**
- **Matplotlib** (optional for plotting, not shown in repo yet)
