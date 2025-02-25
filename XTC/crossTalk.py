import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


# Constants
SAMPLE_RATE = 44100  # 44.1 kHz sampling rate
DURATION = 5  # Duration in seconds
CHANNELS = 2  # Stereo microphone

# Function to apply bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=300, highcut=3400, fs=SAMPLE_RATE, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# Record audio
def record_audio(duration, fs, channels):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype=np.float32)
    sd.wait()
    print("Recording complete.")
    return audio

# Perform ICA for crosstalk cancellation
def apply_ica(audio):
    print("Applying ICA for crosstalk cancellation...")
    ica = FastICA(n_components=2)
    separated_sources = ica.fit_transform(audio)  # Separate sources
    return separated_sources

# Playback function
def play_audio(audio, fs):
    print("Playing processed audio...")
    sd.play(audio, samplerate=fs)
    sd.wait()
    print("Playback complete.")

if __name__ == "__main__":
    # Record
    recorded_audio = record_audio(DURATION, SAMPLE_RATE, CHANNELS)
    
    # Apply bandpass filter
    filtered_audio = np.apply_along_axis(bandpass_filter, 0, recorded_audio)
    
    # Apply ICA
    processed_audio = apply_ica(filtered_audio)
    
    # Normalize audio before playback
    processed_audio = processed_audio / np.max(np.abs(processed_audio))
    
    # Play separated sources
    play_audio(processed_audio, SAMPLE_RATE)