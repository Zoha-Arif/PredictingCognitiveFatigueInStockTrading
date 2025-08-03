import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load preprocessed data
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Wrap in DataLoaders
train_data = torch.utils.data.TensorDataset(X_train, y_train)
val_data = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)

# Define the LSTM model
class FatigueLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take the output of the last time step
        out = self.fc(out)
        return self.sigmoid(out)

# Initialize model
model = FatigueLSTM(input_size=X.shape[2], hidden_size=64, num_layers=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

# Evaluate
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        preds = model(batch_x).squeeze().round()
        all_preds.extend(preds.numpy())
        all_labels.extend(batch_y.numpy())

print("\n📊 Evaluation on Validation Set:")
print(classification_report(all_labels, all_preds, digits=3))

torch.save(model.state_dict(), "fatigue_model.pt")