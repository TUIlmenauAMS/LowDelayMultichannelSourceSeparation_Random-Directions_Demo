import torch
import torch.nn as nn
import torchaudio

Datasetwaveform, sample_rate = torchaudio.load("training_data.wav")

# Define LSTM Model
class LSTM_XTC(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(LSTM_XTC, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc(h)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM_XTC().to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert waveform to training format
input_data = Datasetwaveform.T.to(device)
target_data = Datasetwaveform.T.to(device)  # Assume training with clean reference

# Train Model
for epoch in range(100):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target_data)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save Model
torch.save(model.state_dict(), "xtc_model.pth")
print("✅ Model Training Complete!")
