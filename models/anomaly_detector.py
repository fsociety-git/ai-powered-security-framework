import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16))
        self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_anomaly_model(data_path):
    df = pd.read_csv(data_path)
    data = df.values.astype(float)
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    data = torch.tensor(data, dtype=torch.float32)

    model = Autoencoder(data.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # Simplified
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'anomaly_model.pth')
    return model

def detect_anomaly(input_data):
    model = Autoencoder(10)  # Adjust dim
    model.load_state_dict(torch.load('anomaly_model.pth'))
    model.eval()
    input_tensor = torch.tensor(np.array([float(x) for x in input_data.split(',')]), dtype=torch.float32).unsqueeze(0)
    output = model(input_tensor)
    loss = nn.MSELoss()(output, input_tensor)
    return {"anomaly": loss.item() > 0.1}  # Threshold
