import numpy as np
import sounddevice as sd
import queue
from scipy.signal import butter, lfilter

# 🎛 Constants
SAMPLE_RATE = 44100  # Audio sample rate (44.1 kHz)
CHANNELS = 2  # Stereo input (left & right)
BUFFER_SIZE = 2048  # Increased buffer size to avoid buffer overflows
LMS_FILTER_ORDER = 32  # Order of the adaptive filter
LMS_LEARNING_RATE = 0.01  # Adaptive filter learning rate

# Queue for handling real-time audio
audio_queue = queue.Queue()

# 🛠 Bandpass Filter to Remove Noise
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=300, highcut=3400, fs=SAMPLE_RATE, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# 🎯 LMS Adaptive Crosstalk Cancellation Filter
class LMSFilter:
    def __init__(self, filter_order=LMS_FILTER_ORDER, mu=LMS_LEARNING_RATE):
        self.order = filter_order
        self.mu = mu
        self.weights = np.zeros(filter_order)

    def process(self, reference, signal):
        """Adaptive LMS filtering to remove crosstalk."""
        n = len(signal)
        output = np.zeros(n)

        for i in range(self.order, n):
            x = reference[i - self.order:i]  # Reference signal window
            y = np.dot(self.weights, x)  # Compute filter output
            e = signal[i] - y  # Error signal (desired - estimated)
            self.weights += self.mu * e * x  # LMS weight update
            output[i] = e  # Store filtered signal

        return output

# 🎤 Real-Time Audio Processing Callback
def audio_callback(indata, outdata, frames, time, status):
    """Handles real-time audio recording and playback."""
    if status:
        print(f"🔴 Sounddevice Error: {status}")

    try:
        # Ensure buffer doesn't overflow
        if not audio_queue.full():
            audio_queue.put_nowait(indata.copy())

        # Fetch latest audio chunk
        audio_data = audio_queue.get_nowait()

        # Split left and right channels
        left_channel = audio_data[:, 0]
        right_channel = audio_data[:, 1]

        # Apply bandpass filter
        left_filtered = bandpass_filter(left_channel)
        right_filtered = bandpass_filter(right_channel)

        # Apply LMS Crosstalk Cancellation
        left_processed = lms_left.process(right_filtered, left_filtered)
        right_processed = lms_right.process(left_filtered, right_filtered)

        # Normalize output to avoid clipping
        left_processed = left_processed / (np.max(np.abs(left_processed)) + 1e-9)
        right_processed = right_processed / (np.max(np.abs(right_processed)) + 1e-9)

        # Combine processed audio into stereo output
        outdata[:, 0] = left_processed
        outdata[:, 1] = right_processed

    except queue.Empty:
        print("⚠️ Queue empty, skipping processing")
        outdata[:] = 0  # Prevent underflow by sending silent frames

# 🔥 Start Real-Time Processing
def start_real_time_xtc():
    global lms_left, lms_right

    # Initialize LMS filters for left and right channels
    lms_left = LMSFilter()
    lms_right = LMSFilter()

    print("🎙️ Real-Time Crosstalk Cancellation Running... Press Ctrl+C to Stop.")

    try:
        # Open input-output stream for real-time processing
        with sd.Stream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32,
                       blocksize=BUFFER_SIZE, callback=audio_callback):
            while True:
                sd.sleep(10)  # Avoid CPU overuse
    except KeyboardInterrupt:
        print("🛑 Stopping Real-Time Processing...")
    except Exception as e:
        print(f"🔴 Unexpected Error: {e}")

# 🔥 Run the real-time crosstalk cancellation
if __name__ == "__main__":
    start_real_time_xtc()
