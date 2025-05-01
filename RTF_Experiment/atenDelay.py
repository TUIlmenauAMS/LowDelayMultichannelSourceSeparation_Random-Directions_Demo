import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Load HRIR files
HRIR_neg45_R, sr1 = sf.read(r"RTF_Experiment\HRIR_neg45_R.wav")
HRIR_neg45_L, sr2 = sf.read(r"RTF_Experiment\HRIR_neg45_L.wav")
HRIR_pos45_R, sr3 = sf.read(r"RTF_Experiment\HRIR_pos45_R.wav")
HRIR_pos45_L, sr4 = sf.read(r"RTF_Experiment\HRIR_pos45_L.wav")

# Ensure mono signals (use only one channel if stereo)
def to_mono(signal):
    return signal[:, 0] if signal.ndim > 1 else signal

HRIR_neg45_R = to_mono(HRIR_neg45_R)
HRIR_neg45_L = to_mono(HRIR_neg45_L)
HRIR_pos45_R = to_mono(HRIR_pos45_R)
HRIR_pos45_L = to_mono(HRIR_pos45_L)

# Use sample rate from any (assuming same for all)
sr = sr3
N = len(HRIR_pos45_R)

# FFT to get HRTF
HRTF_pos45_R = np.fft.fft(HRIR_pos45_R)
HRTF_pos45_L = np.fft.fft(HRIR_pos45_L)

# Only keep positive frequencies
HRTF_pos45_R = HRTF_pos45_R[:N // 2]
HRTF_pos45_L = HRTF_pos45_L[:N // 2]

# Compute magnitude (in dB)
HRTF_pos45_R_mag = 10 * np.log10(np.abs(HRTF_pos45_R) + 1e-12)
HRTF_pos45_L_mag = 10 * np.log10(np.abs(HRTF_pos45_L) + 1e-12)

# Compute RTF
RTF_complex = HRTF_pos45_R / (HRTF_pos45_L + 1e-12)

# Attenuation in dB
attenuation_dB = 20 * np.log10(np.abs(RTF_complex) + 1e-12)

# Phase difference and delay (in seconds)
phase_diff = np.angle(RTF_complex)
frequencies = np.fft.fftfreq(N, d=1/sr)[:N // 2]
frequencies[frequencies == 0] = 1e-6  # avoid divide-by-zero
delay_seconds = np.unwrap(phase_diff) / (2 * np.pi * frequencies)

# Plotting
plt.figure(figsize=(12, 8))

# Plot Attenuation
plt.subplot(3, 1, 1)
plt.plot(frequencies, attenuation_dB, 'r')
plt.title("Attenuation (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("dB")
plt.grid(True)

# Plot RTF Magnitude
plt.subplot(3, 1, 2)
plt.plot(frequencies, np.abs(RTF_complex), 'g')
plt.title("RTF Magnitude")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(frequencies, delay_seconds, 'b')
plt.title("Estimated Delay Between Microphones")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Delay (seconds)")
plt.grid(True)
plt.tight_layout() 
plt.show()
print("DONE")