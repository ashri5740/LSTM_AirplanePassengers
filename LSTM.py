# ============================================
# LSTM Time Series Prediction using PyTorch
# Dataset: Airline Passengers Dataset
# ============================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================
# Load Dataset
# ============================================

# CSV file should contain:
# Month,Passengers

df = pd.read_csv("AirPassengers.csv")

print("First 5 Rows:")
print(df.head())

# ============================================
# Data Preprocessing
# ============================================

# Convert Month column to datetime
# Load dataset
df = pd.read_csv("AirPassengers.csv")

# Print column names
print(df.columns)

# Convert Month column
df['Month'] = pd.to_datetime(df['Month'])

# Sort values
df = df.sort_values('Month')

# Select passenger column
data = df['#Passengers'].to_numpy().reshape(-1, 1)

# Normalize
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# ============================================
# Create Sequences
# ============================================

def create_sequences(data, sequence_length):

    X = []
    y = []

    for i in range(len(data) - sequence_length):

        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    return np.array(X), np.array(y)

sequence_length = 10

X, y = create_sequences(scaled_data, sequence_length)

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

# ============================================
# Train-Test Split
# ============================================

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# ============================================
# Convert to PyTorch Tensors
# ============================================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False
)

# ============================================
# Build LSTM Model
# ============================================

class LSTMModel(nn.Module):

    def __init__(self, input_size=1, hidden_size=64, num_layers=2):

        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # Initialize hidden state
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        )

        # Initialize cell state
        c0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        )

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get output from last timestep
        out = out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(out)

        return out

# Create model
model = LSTMModel()

print("\nModel Architecture:")
print(model)

# ============================================
# Loss Function and Optimizer
# ============================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ============================================
# Train the Model
# ============================================

epochs = 100

train_losses = []

print("\nTraining Started...\n")

for epoch in range(epochs):

    model.train()

    epoch_loss = 0

    for X_batch, y_batch in train_loader:

        # Forward pass
        outputs = model(X_batch)

        # Calculate loss
        loss = criterion(outputs, y_batch)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

print("\nTraining Completed!")

# ============================================
# Evaluate the Model
# ============================================

model.eval()

with torch.no_grad():

    predictions = model(X_test)

# Convert to numpy
predictions = predictions.numpy()
y_actual = y_test.numpy()

# Inverse transform
predictions = scaler.inverse_transform(predictions)
y_actual = scaler.inverse_transform(y_actual)

# ============================================
# Calculate Metrics
# ============================================

mse = mean_squared_error(y_actual, predictions)

rmse = np.sqrt(mse)

print("\nEvaluation Metrics")
print("-------------------")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# ============================================
# Plot Training Loss
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(train_losses)

plt.title("Training Loss Curve")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.show()

# ============================================
# Plot Actual vs Predicted Values
# ============================================

plt.figure(figsize=(12, 6))

plt.plot(y_actual, label='Actual Values')

plt.plot(predictions, label='Predicted Values')

plt.title("Actual vs Predicted Passenger Values")

plt.xlabel("Time")

plt.ylabel("Passengers")

plt.legend()

plt.grid(True)

plt.show()

# ============================================
# Predict Future Value
# ============================================

last_sequence = scaled_data[-sequence_length:]

last_sequence = torch.tensor(
    last_sequence.reshape(1, sequence_length, 1),
    dtype=torch.float32
)

model.eval()

with torch.no_grad():

    future_prediction = model(last_sequence)

future_prediction = future_prediction.numpy()

future_prediction = scaler.inverse_transform(future_prediction)

print("\nNext Predicted Passenger Value:")
print(future_prediction[0][0])

# ============================================
# Hyperparameter Tuning Suggestions
# ============================================

"""
Try changing:

1. hidden_size = 32, 64, 128
2. num_layers = 1, 2, 3
3. learning_rate = 0.01, 0.001
4. sequence_length = 5, 10, 20
5. epochs = 50, 100, 200

These can improve prediction accuracy.
"""

# ============================================
# End of Program
# ============================================