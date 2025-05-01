import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the audio files
HRIR_neg45_R, sr1 = sf.read(r"RTF_Experiment\HRIR_neg45_R.wav")
HRIR_neg45_L, sr2 = sf.read(r"RTF_Experiment\HRIR_neg45_L.wav")
HRIR_pos45_R, sr3 = sf.read(r"RTF_Experiment\HRIR_pos45_R.wav")
HRIR_pos45_L, sr4 = sf.read(r"RTF_Experiment\HRIR_pos45_L.wav")

# Create time axes
t1 = [i / sr1 for i in range(len(HRIR_neg45_R))]
t2 = [i / sr2 for i in range(len(HRIR_neg45_L))]
t3 = [i / sr3 for i in range(len(HRIR_pos45_R))]
t4 = [i / sr4 for i in range(len(HRIR_pos45_L))]


HRTF_pos45_R = np.abs(np.fft.fft(HRIR_pos45_R))
HRTF_pos45_L = np.abs(np.fft.fft(HRIR_pos45_L))
HRTF_pos45_R = HRTF_pos45_R[:len(HRTF_pos45_R) // 2]
HRTF_pos45_L = HRTF_pos45_L[:len(HRTF_pos45_L) // 2]
HRTF_pos45_R_db = 10 * np.log10(HRTF_pos45_R + 1e-12)  # Avoid log(0)
HRTF_pos45_L_db = 10 * np.log10(HRTF_pos45_L + 1e-12)

RTF = HRTF_pos45_R / (HRTF_pos45_L + 1e-12)  # Avoid division by zero
magnitude_RTF = 20 * np.log10(np.abs(RTF) + 1e-12)  # Avoid log(0)
RRIR = np.abs(np.fft.ifftshift(RTF))

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t1, HRIR_neg45_R, label='HRIR_neg45_R')
plt.plot(t2, HRIR_neg45_L, label='HRIR_neg45_L')
plt.title("RIR +45°")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t3, HRIR_pos45_R, label='Microphone 1')
plt.plot(t4, HRIR_pos45_L, label='Microphone 2')
plt.title("RIR -45°")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)


plt.subplot(2, 2, 3)
plt.plot(magnitude_RTF, 'r', label='RTF_M')
plt.title("Magnitude (dB) of RTF")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude (dB)")
plt.legend()
#plt.xscale('log')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(RRIR, label='RRIR')
plt.title("RRIR")
plt.xlabel("Time [Samples]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
print("RTF plot generated successfully.")
