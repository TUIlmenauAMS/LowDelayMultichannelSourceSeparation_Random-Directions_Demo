import sounddevice as sd
import numpy as np
import soundfile as sf
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import threading

# 🎧 Load two different sound sources (e.g., speech or music files)
source1, sr1 = sf.read("source1.wav")  # First speaker's audio
source2, sr2 = sf.read("source2.wav")  # Second speaker's audio

# Ensure both sources have the same sample rate
assert sr1 == sr2, "Sample rates of both sources must be the same!"
fs = sr1  # Use this sample rate

# Set recording parameters
duration = min(len(source1), len(source2)) / fs  # Set duration to match shortest source
channels = 2  # Two-channel recording

# Normalize sources to prevent clipping
source1 = source1 / np.max(np.abs(source1))
source2 = source2 / np.max(np.abs(source2))

# Combine into stereo output (source1 → Left speaker, source2 → Right speaker)
stereo_output = np.column_stack((source1, source2))

# Function to play the stereo sources
def play_audio():
    print("🔊 Playing two audio sources...")
    sd.play(stereo_output, samplerate=fs)
    sd.wait()

# Function to record from two-channel microphone
def record_audio():
    print("🎤 Recording from two microphones...")
    recording = sd.rec(int(fs * duration), samplerate=fs, channels=channels, dtype=np.float32)
    sd.wait()
    print("✅ Recording complete!")
    
    # Save recorded mixture as a WAV file
    recorded_filename = "two_speakers_recording.wav"
    wav.write(recorded_filename, fs, (recording * 32767).astype(np.int16))  # Convert to 16-bit PCM
    print(f"💾 Recorded audio saved as {recorded_filename}")

    # Plot the recorded waveforms
    time = np.linspace(0, duration, num=recording.shape[0])

    plt.figure(figsize=(10, 5))

    # Plot Channel 1 (Mic 1)
    plt.subplot(2, 1, 1)
    plt.plot(time, recording[:, 0], color='b', label="Microphone 1 (Speaker 1)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform of Microphone 1")
    plt.legend()
    plt.grid()

    # Plot Channel 2 (Mic 2)
    plt.subplot(2, 1, 2)
    plt.plot(time, recording[:, 1], color='r', label="Microphone 2 (Speaker 2)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform of Microphone 2")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Run playback and recording in parallel
play_thread = threading.Thread(target=play_audio)
record_thread = threading.Thread(target=record_audio)

play_thread.start()
record_thread.start()

play_thread.join()
record_thread.join()
