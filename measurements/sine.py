# generate_sine_sweep.py
import numpy as np
from scipy.io.wavfile import write

def generate_sine_sweep(duration, sample_rate, f0, f1):
    """
    Generate a sine sweep (chirp) signal.
    
    Parameters:
        duration (float): Duration of the sweep in seconds.
        sample_rate (int): Sampling rate in Hz.
        f0 (float): Start frequency of the sweep in Hz.
        f1 (float): End frequency of the sweep in Hz.
    
    Returns:
        sweep (np.array): Sine sweep signal.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    phase = 2 * np.pi * f0 * duration / np.log(f1 / f0) * (np.exp(t * np.log(f1 / f0) / duration) - 1)
    sweep = np.sin(phase)
    return sweep

# Parameters for the sine sweep
duration = 10  # Duration of the sweep in seconds
sample_rate = 44100  # Sampling rate in Hz
f0 = 20  # Start frequency in Hz
f1 = 20000  # End frequency in Hz

# Generate the sine sweep
sine_sweep = generate_sine_sweep(duration, sample_rate, f0, f1)

# Normalize the sweep to prevent clipping
sine_sweep = sine_sweep / np.max(np.abs(sine_sweep))

# Generate the inverse sweep (reverse in time)
inverse_sweep = sine_sweep[::-1]  # Reverse the array

# Save the sine sweep as a WAV file
sine_sweep_filename = "sine_sweep.wav"
write(sine_sweep_filename, sample_rate, (sine_sweep * 32767).astype(np.int16))  # Convert to 16-bit PCM
print(f"💾 Sine sweep saved as {sine_sweep_filename}")

# Save the inverse sweep as a WAV file
inverse_sweep_filename = "inverse_sweep.wav"
write(inverse_sweep_filename, sample_rate, (inverse_sweep * 32767).astype(np.int16))  # Convert to 16-bit PCM
print(f"💾 Inverse sweep saved as {inverse_sweep_filename}")