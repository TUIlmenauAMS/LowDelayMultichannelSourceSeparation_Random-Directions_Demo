import sounddevice as sd
import numpy as np
import wave

SAMPLE_RATE = 44100
CHANNELS = 2
DURATION = 5  # Record 10 seconds

print("🎙️ Recording...")
recorded_audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32)
sd.wait()
print("✅ Recording Complete!")

# Save audio file
with wave.open("training_data.wav", "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes((recorded_audio * 32767).astype(np.int16).tobytes())
