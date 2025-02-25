import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# Constants
SAMPLE_RATE = 44100  # 44.1 kHz sampling rate
DURATION = 5  # Duration in seconds
CHANNELS = 2  # Stereo microphone

# Function to apply bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=300, highcut=3400, fs=SAMPLE_RATE, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# LMS Adaptive Filter for Crosstalk Cancellation
def lms_filter(reference, signal, mu=0.01, filter_order=32):
    n = len(signal)
    w = np.zeros(filter_order)  # Initialize filter weights
    output = np.zeros(n)
    
    for i in range(filter_order, n):
        x = reference[i-filter_order:i]  # Reference signal segment
        y = np.dot(w, x)  # Compute the filter's output (weighted sum)
        e = signal[i] - y  # Error: difference between actual and estimated signal
        w += mu * e * x  # Update filter weights
        output[i] = e  # Store error signal as the cleaned signal
        
    return output

# Record audio
def record_audio(duration, fs, channels):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype=np.float32)
    sd.wait()
    print("Recording complete.")
    return audio

# Playback function
def play_audio(audio, fs):
    print("Playing processed audio...")
    sd.play(audio, samplerate=fs)
    sd.wait()
    print("Playback complete.")

if __name__ == "__main__":
    # Record
    recorded_audio = record_audio(DURATION, SAMPLE_RATE, CHANNELS)
    
    # Apply bandpass filter to each channel separately
    filtered_audio = np.apply_along_axis(bandpass_filter, 0, recorded_audio)
    
    # Separate channels
    left_channel = filtered_audio[:, 0]
    right_channel = filtered_audio[:, 1]
    
    # Apply LMS Filter to reduce crosstalk
    left_processed = lms_filter(right_channel, left_channel)
    right_processed = lms_filter(left_channel, right_channel)
    
    # Combine processed channels
    processed_audio = np.column_stack((left_processed, right_processed))
    
    # Normalize audio before playback
    processed_audio = processed_audio / np.max(np.abs(processed_audio))
    
    # Play the processed audio
    play_audio(processed_audio, SAMPLE_RATE)
    
    # Create time axis for plotting
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    
    # Plot the original filtered signals vs. the processed signals
    plt.figure(figsize=(14, 8))
    
    # Plot original left channel
    plt.subplot(2, 2, 1)
    plt.plot(t, left_channel, color='blue')
    plt.title('Original Left Channel (Filtered)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    # Plot processed left channel
    plt.subplot(2, 2, 2)
    plt.plot(t, left_processed, color='red')
    plt.title('Processed Left Channel (Crosstalk Canceled)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    # Plot original right channel
    plt.subplot(2, 2, 3)
    plt.plot(t, right_channel, color='blue')
    plt.title('Original Right Channel (Filtered)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    # Plot processed right channel
    plt.subplot(2, 2, 4)
    plt.plot(t, right_processed, color='red')
    plt.title('Processed Right Channel (Crosstalk Canceled)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
