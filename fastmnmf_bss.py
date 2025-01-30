#FastMNMF multichannel source separation from pyroomacoustics
#Introduces at least delay of hopsize = fft_size//2, or 2048 samples
#Gerald Schuller, Aug. 5, 2021

import numpy as np
import scipy.io.wavfile as wav
import time
import os
import matplotlib.pyplot as plt
#!pip3 install pyroomacoustics==0.7.3
#from pyroomacoustics.bss.trinicon import trinicon


def playsound(audio, samplingRate, channels):
    #funtion to play back an audio signal, in array "audio"
    import pyaudio
    p = pyaudio.PyAudio()
    # open audio stream

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplingRate,
                    output=True)
    
    audio=np.clip(audio,-2**15,2**15-1)
    sound = (audio.astype(np.int16).tostring())
    #sound = (audio.astype(np.int16).tostring())ICAabskl_puredelay_lowpass.py
    stream.write(sound)

    # close stream and terminate audio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    return  

def separation_fastmnmf(mixfile, plot=True):
   #Separates 2 audio sources from the multichannel mix in the mixfile,
   #Using the FastMNMF method.
   #plot=True plots the resulting unmixed wave forms.
   
   
   import scipy.io.wavfile as wav
   import scipy.optimize as opt
   import os
   import time
   import matplotlib.pyplot as plt
   import pyroomacoustics as pra
   
   samplerate, X = wav.read(mixfile)
   print("X.shape=", X.shape)
   X=X*1.0/np.max(abs(X)) #normalize

   starttime=time.time()

   # STFT analysis parameters
   fft_size = 4096  # `fft_size / fs` should be ~RT60
   hop = fft_size // 2  # half-overlap
   win_a = pra.hann(fft_size)  # analysis window
   # optimal synthesis window
   win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

   # STFT
   # audio.shape == (nsamples, nchannels)
   # X.shape == (nframes, nfrequencies, nchannels)
   X = pra.transform.stft.analysis(X, fft_size, hop, win=win_a)

   # Separation
   #Y = pra.bss.auxiva(X, n_iter=20)
   Y = pra.bss.fastmnmf(X,n_src=2)

   # iSTFT (introduces an offset of `hop` samples)
   # X_sep contains the time domain separated signals
   # X_sep.shape == (new_nsamples, nchannels)
   X_sep = pra.transform.stft.synthesis(Y, fft_size, hop, win=win_s)
   X_sep = X_sep[hop:,:]   #compensate for delay of hop size
   print("X_sep.shape=", X_sep.shape)
   #X_sep, demixmat = trinicon(X.T, filter_length=2048, return_filters=True)
   #For shorter filter length it takes longer and separation becomes worse!

   endtime=time.time()
   processingtime=endtime-starttime
   print("Duration of optimization:", endtime-starttime, "sec.")

   #X_sep=X_sep.T
   if plot==True:
      plt.plot(X_sep[:,0])
      plt.plot(X_sep[:,1])
      plt.title('The unmixed channels')
      plt.show()
   wav.write("sepchan_fastmnmf.wav",samplerate,np.int16(np.clip(X_sep*2**15,-2**15,2**15-1)))
   print("Written to sepchan_fastmnmf.wav, play with: play sepchan_fastmnmf.wav remix 1/2")
   return processingtime, X_sep
   

if __name__ == '__main__':

   #Signal to separate:
   #mixfile= "mix16000ampcubetones.wav"
   #mixfile = "mix16000stereo_espeechf_espeechm.wav"
   #mixfile = "mix16000.wav"
   #mixfile="stereomusicnoise.wav"
   mixfile="musicnoiselivingroom.wav"
   processingtime, X_sep=separation_fastmnmf(mixfile, plot=True)

   samplerate, X = wav.read(mixfile)
   print("samplerate=", samplerate)
   chanout=2
   X_sep=X_sep*1.0/np.max(abs(X_sep))
   for c in range(chanout):
      os.system('espeak -s 120 "Separated Channel'+str(c)+' " ')
      playsound(X_sep[:,c]*2**15, samplerate, 1)
   
   print("X_sep.shape=", X_sep.shape)
   print("np.max(X_sep)=", np.max(X_sep))
   


