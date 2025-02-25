import numpy as np
import scipy.io.wavfile as wav
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

rate, mics = wav.read("mix16000.wav")  #

mic1 = mics[:,0]  # Microphone 1 input
mic2 = mics[:,1]  # Microphone 2 input

# assert rate1 == rate2, "Sampling rates do not match!"

# Normalize and create the mixture matrix
mic1 = mic1 / np.max(np.abs(mic1))  # Normalize
mic2 = mic2 / np.max(np.abs(mic2))  # Normalize
M = np.c_[mic1, mic2[:len(mic1)]]  # Shape: (samples, 2)

# Apply FastICA for source separation
ica = FastICA(n_components=2, random_state=42)
S_separated = ica.fit_transform(M)  # Extract sources

# Normalize separated signals
S_separated = S_separated / np.max(np.abs(S_separated), axis=0)

# Save the separated signals as audio files
wav.write("source1.wav", rate, (S_separated[:, 0] * 32767).astype(np.int16))
wav.write("source2.wav", rate, (S_separated[:, 1] * 32767).astype(np.int16))

print("Source separation completed! Check 'source1.wav' and 'source2.wav'")

# Plot the original microphone signals and separated sources
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
axs[0].plot(M[:, 0], color="blue")
axs[0].set_title("Microphone 1 Recording")
axs[1].plot(M[:, 1], color="red")
axs[1].set_title("Microphone 2 Recording")
axs[2].plot(S_separated[:, 0], color="green")
axs[2].set_title("Separated Source 1")
axs[3].plot(S_separated[:, 1], color="purple")
axs[3].set_title("Separated Source 2")
plt.tight_layout()
plt.show()




from sklearn.decomposition import PCA

# Perform PCA whitening
pca = PCA(n_components=2, whiten=True)
M_whitened = pca.fit_transform(M)

# Apply ICA after PCA
S_separated = ica.fit_transform(M_whitened)

import librosa
import librosa.display

# Convert signals to frequency domain using STFT
D1 = librosa.stft(mic1, n_fft=2048, hop_length=512)
D2 = librosa.stft(mic2, n_fft=2048, hop_length=512)

# Combine STFTs for ICA
M_freq = np.vstack([np.abs(D1).flatten(), np.abs(D2).flatten()]).T  # Only magnitude

# Apply ICA in frequency domain
S_separated_freq = ica.fit_transform(M_freq)

# Convert back to time domain
S1_time = librosa.istft(S_separated_freq[:, 0].reshape(D1.shape), hop_length=512)
S2_time = librosa.istft(S_separated_freq[:, 1].reshape(D2.shape), hop_length=512)

# Save separated sources
wav.write("source1_time.wav", rate, (S1_time * 32767).astype(np.int16))
wav.write("source2_time.wav", rate, (S2_time * 32767).astype(np.int16))

from pykalman import KalmanFilter

# Define Kalman Filter for adaptive mixing matrix estimation
kf = KalmanFilter(n_dim_obs=2, n_dim_state=2, initial_state_mean=[0, 0])
print("DONE KalmanFilter")

# Fit the model to learn the optimal mixing coefficients
kf = kf.em(M, n_iter=10)
print("DONE kf")

M_filtered, _ = kf.filter(M)
print("DONE M_filtered")

# Apply ICA after filtering
S_separated = ica.fit_transform(M_filtered)

print("DONE")

fig, axs = plt.subplots(4, 1, figsize=(10, 8))
axs[0].plot(M[:, 0], color="blue")
axs[0].set_title("Microphone 1 Recording")
axs[1].plot(M[:, 1], color="red")
axs[1].set_title("Microphone 2 Recording")
axs[2].plot(S_separated[:, 0], color="green")
axs[2].set_title("Separated Source 1")
axs[3].plot(S_separated[:, 1], color="purple")
axs[3].set_title("Separated Source 2")
plt.tight_layout()
plt.show()

S_separated = S_separated / np.max(np.abs(S_separated), axis=0)

# Save the separated signals as audio files
wav.write("source11.wav", rate, (S_separated[:, 0] * 32767).astype(np.int16))
wav.write("source22.wav", rate, (S_separated[:, 1] * 32767).astype(np.int16))

print("Source separation completed! Check 'source1.wav' and 'source2.wav'")






