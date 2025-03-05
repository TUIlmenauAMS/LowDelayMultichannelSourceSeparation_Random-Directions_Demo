# play_and_record_rir.py
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import read, write
import threading
import matplotlib.pyplot as plt


# Load the sine sweep and inverse sweep from the WAV files
sine_sweep_sample_rate, sine_sweep = read("sine_sweep.wav")
inverse_sweep_sample_rate, inverse_sweep = read("inverse_sweep.wav")

# Ensure the sample rates match
assert sine_sweep_sample_rate == inverse_sweep_sample_rate, "Sample rates of the sweeps must match!"
sample_rate = sine_sweep_sample_rate

# Normalize the sweeps (WAV files are loaded as integers, so convert to float)
sine_sweep = sine_sweep.astype(np.float32) / 32767.0
inverse_sweep = inverse_sweep.astype(np.float32) / 32767.0

# Play the sine sweep through both speakers (left and right channels)
stereo_output_sine_sineInverse = np.column_stack((sine_sweep, inverse_sweep)) 
stereo_output_sin= np.column_stack((sine_sweep, sine_sweep))  
stereo_output_inverse = np.column_stack((inverse_sweep, inverse_sweep))  
rec_duration=10


# Function to play the both sweep
def playBoth():
    print("🔊 Playing sine sweep on Left & inverse sine sweep on Right speaker...")
    sd.play(stereo_output_sine_sineInverse, samplerate=sample_rate)
    sd.wait()




# Function to play the sin sweep
def play_sweep_on_both_speaker():
    print("🔊 Playing inverse sweep through both speakers...")
    sd.play(stereo_output_sin, samplerate=sample_rate)
    sd.wait()



# Function to play the inverse sweep
def play_inverse_sweep_on_both_speaker():
    print("🔊 Playing inverse sweep through both speakers...")
    sd.play(stereo_output_inverse, samplerate=sample_rate)
    sd.wait()




# Function to record the RIR
def record_rir():
    print("🎤 Recording RIR...")
    print(sd.query_devices())
    recording = sd.rec(int(sample_rate * rec_duration), samplerate=sample_rate, channels=2, dtype=np.float32)
    sd.wait()
    print("✅ Recording complete!")
    
    # Save the recorded RIR as a WAV file
    rir_filename = "rir_recording_both_speakers.wav"
    write(rir_filename, sample_rate, (recording * 32767).astype(np.int16))  # Convert to 16-bit PCM
    print(f"💾 RIR saved as {rir_filename}")
    plot_recorded_rir(recording, sample_rate)




def plot_recorded_rir(recording, sample_rate):
    time = np.linspace(0, rec_duration, num=recording.shape[0])

    plt.figure(figsize=(10, 6))

    # Plot Channel 1 (Left Speaker)
    plt.subplot(2, 1, 1)
    plt.plot(time, recording[:, 0], color='b', label="Left Channel (Speaker 1)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform of Left Channel")
    plt.legend()
    plt.grid()

    # Plot Channel 2 (Right Speaker)
    plt.subplot(2, 1, 2)
    plt.plot(time, recording[:, 1], color='r', label="Right Channel (Speaker 2)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform of Right Channel")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()





play_thread = threading.Thread(target=playBoth)  # Change to play_inverse_sweep for the inverse sweep
record_thread = threading.Thread(target=record_rir)

play_thread.start()
record_thread.start()

play_thread.join()
record_thread.join()