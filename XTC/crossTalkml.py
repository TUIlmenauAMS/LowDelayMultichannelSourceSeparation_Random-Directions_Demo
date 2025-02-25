import numpy as np
import sounddevice as sd
import queue
import torch
import torch.nn as nn
import torchaudio.transforms as T

# 🎛 Constants
SAMPLE_RATE = 44100  # 44.1 kHz sample rate
CHANNELS = 2  # Stereo input
BUFFER_SIZE = 2048  # Increased buffer size to avoid overflow
SEQ_LEN = 128  # Sequence length for LSTM model

# Queue to handle real-time audio streaming
audio_queue = queue.Queue()

# 🎯 Deep Learning Model for Crosstalk Cancellation
class LSTM_XTC(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(LSTM_XTC, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc(h)
        return out

# 🎯 Load Pretrained Model (Assume it's saved as "xtc_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_XTC().to(device)
model.load_state_dict(torch.load("xtc_model.pth", map_location=device))
model.eval()

# 🔥 Real-Time Audio Processing Callback
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(f"🔴 Sounddevice Error: {status}")

    try:
        # Ensure buffer doesn't overflow
        if not audio_queue.full():
            audio_queue.put_nowait(indata.copy())

        # Fetch latest audio chunk
        audio_data = audio_queue.get_nowait()

        # Normalize input
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-9)

        # Convert to PyTorch tensor
        input_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(device)

        # Apply ML-Based XTC
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Convert back to NumPy
        processed_audio = output_tensor.squeeze(0).cpu().numpy()

        # Normalize output to prevent clipping
        processed_audio = processed_audio / (np.max(np.abs(processed_audio)) + 1e-9)

        # Output processed audio
        outdata[:, 0] = processed_audio[:, 0]  # Left channel
        outdata[:, 1] = processed_audio[:, 1]  # Right channel

    except queue.Empty:
        print("⚠️ Queue empty, skipping processing")
        outdata[:] = 0  # Prevent underflow by sending silent frames

# 🔥 Start Real-Time Processing
def start_real_time_xtc():
    print("🎙️ Machine Learning-Based Crosstalk Cancellation Running... Press Ctrl+C to Stop.")

    try:
        with sd.Stream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32,
                       blocksize=BUFFER_SIZE, callback=audio_callback):
            while True:
                sd.sleep(10)  # Allow CPU to idle
    except KeyboardInterrupt:
        print("🛑 Stopping Real-Time Processing...")

# 🔥 Run the real-time crosstalk cancellation
if __name__ == "__main__":
    start_real_time_xtc()
