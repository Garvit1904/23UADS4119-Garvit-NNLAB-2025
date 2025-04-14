# RNN Time Series Prediction using PyTorch

## Objective

To build and train a Recurrent Neural Network (RNN) using PyTorch for predicting the next value in a time series, specifically using a sine wave as sample data.

---

## Description of the Model

This model is a simple **RNN-based regressor** designed to handle time series data. The network architecture includes:

- **Input Layer**: Takes a sequence of sine wave values.
- **RNN Layer**: One RNN layer with 50 hidden units.
- **Output Layer**: A fully connected layer that outputs the next value in the sequence.

The model uses **Mean Squared Error (MSE)** as the loss function and **Adam optimizer** for training.

---

##  Python Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def generate_data(seq_length=50):
    x = np.linspace(0, 100, 1000)
    data = np.sin(x)
    return data

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x_seq = data[i:i+seq_length]
        y_seq = data[i+seq_length]
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs), np.array(ys)

class RNNPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(RNNPredictor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  
        out = self.fc(out)
        return out


data = generate_data()
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

seq_length = 20
X, y = create_sequences(data, seq_length)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

model = RNNPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    print(f"\nTest Loss: {test_loss.item():.4f}")

plt.plot(y_test.numpy(), label='True')
plt.plot(predictions.numpy(), label='Predicted')
plt.legend()
plt.title("RNN Time Series Prediction")
plt.show()
```

---

## Description of Code

- **Data Generation**: A sine wave is used to simulate time series data.
- **Preprocessing**: Data is normalized and split into sequences.
- **Model Architecture**: Basic RNN followed by a fully connected layer.
- **Training Loop**: Optimizes the model using MSE loss.
- **Evaluation**: Calculates test loss and visualizes predictions.

---

## Performance Evaluation



The model predictions (orange line) match very closely with the actual sine wave values (blue line), indicating strong learning performance.

- **Final Training Loss**: Approached 0.0000
- **Test Loss**: Very low, indicating high generalization ability

---

## My Comments

- Not suitable for long-range dependencies — consider using **LSTM** or **GRU**.
- Model assumes clean sine wave data — might not perform as well on noisy or irregular data.


