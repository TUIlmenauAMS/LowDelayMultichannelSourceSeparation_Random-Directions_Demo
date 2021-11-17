#Trinicon multichannel source separation from pyroomacoustics
#Trinicon: Long FIR filters for demixing
#Gerald Schuller, Aug. 5, 2021

import numpy as np
import scipy.io.wavfile as wav
import time
import os
import matplotlib.pyplot as plt
import scipy.signal


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
    
def unmixing(coeffs, X, chanout, state=[]):
   #Applies an anmixing matrix build from coeffs to a multi-channel signal X,
   #where each column is a channel
   #Arguments: 
   #Coeffs: 3D array which contains the FIR filter coefficents (from, to, fir coeff) in the third dimension
   #X= multichannel signal from the microphones, each column is a channel
   #to chanout: number of signals to separate in the output
   #state = states from previous delay filter run, each column is a filter set
   #state: 3-dim array, a filter set is: state[from,to,:]
   #
   #Returns the resulting (unmixed) stereo signal X_sep
   #Unmixing Process:
   #Delayed and attenuated versions of opposing microphones are subtracted:
   #Xsep[:,tochan]= sum_fromchan lfilter(X[:,fromchan], coeffs[fromchan,tochan,:])
   #...
   print("coeffs.shape=", coeffs.shape)
   chanin=X.shape[1] #number of channels of the input signal

   #print("Channels=", chanin) 
   X_sep=np.zeros((X.shape[0],chanout)) #Shape of the "chanout" channel output signal
  
   #The channels are delayed with the allpass filter:
   Y=np.zeros((X.shape[0], X.shape[1], chanout)) #The delayed signals, shape: signal length, input chan, output chan
   #print("X.shape=", X.shape,"Y.shape=", Y.shape)
   if state!=[]:
      for fromchan in range(chanin):
         for tochan in range(chanout):  #2 channel output
            Y[:,fromchan, tochan], state[fromchan, tochan,:]=scipy.signal.lfilter(coeffs[fromchan,tochan,:], 1 ,X[:,fromchan], zi=state[fromchan, tochan,:])
   else:
      for fromchan in range(chanin):
         for tochan in range(chanout):  #2 channel output
            X_sep[:, tochan]  +=scipy.signal.lfilter( coeffs[fromchan,tochan,:], 1 ,X[:,fromchan])

   
   return X_sep #, state

def separation_trinicon(mixfile, plot=True):
   #Separates 2 audio sources from the multichannel mix in the mixfile,
   #Using the Trinicon method.
   #plot=True plots the resulting unmixed wave forms.

   import scipy.io.wavfile as wav
   import scipy.optimize as opt
   import os
   import time
   import matplotlib.pyplot as plt
   from pyroomacoustics.bss.trinicon import trinicon

   samplerate, X = wav.read(mixfile)
   print("X.shape=", X.shape)
   X=X*1.0/np.max(abs(X)) #normalize

   starttime=time.time()
   blocksize=8000
   #siglen=X.shape[0] #length of signal
   siglen=max(X.shape)
   blocks= int(siglen/blocksize)
   #blocks=0
   blockaccumulator=X[0:blocksize,:]
   blockno=0

   starttime=time.time()
   for ob in range(1): #sub-periods after which to run the optimization, outer blocks, seems best for 1 outer block.
      #Accumulate part of the signal in a signal "accumulator" of size "blocksize" (8000 samples, or 0.5s):
      for i in range(min(blocks,16)): #accumulate audio blocks over about 3 seconds:
         blockaccumulator=0.98*blockaccumulator + 0.02*X[blockno*blocksize+np.arange(blocksize)]
         blockno+=1
            
   chanout=2
   #According to:
   #https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.bss.trinicon.html
   #Trinicon uses filter_length=2048 by default
   #Bss_eval can process time-invariant filter distortion of max filter length of 512
   #hard-coded for 2 output channels:
   X_sep, demixmat = trinicon(blockaccumulator.T, filter_length=512, return_filters=True)
   #For shorter filter length it takes longer and separation becomes worse!
   print("X_sep.shape=", X_sep.shape, "demixmat.shape=", demixmat.shape)



   endtime=time.time()
   processingtime=endtime-starttime
   print("Duration of optimization:", endtime-starttime, "sec.")

   starttime=time.time()
   X_sep=unmixing(demixmat, X, chanout)
   endtime=time.time()
   print("Duration of unmixing:", endtime-starttime, "sec.")
   #X_sep=X_sep.T

   print("np.max(X_sep)=", np.max(X_sep))
   wav.write("sepchan_trinicon_online.wav",samplerate,np.int16(np.clip(X_sep*2**15,-2**15,2**15-1)))
   print("Written to sepchan_trinicon_online.wav, play with: play sepchan_trinicon_online.wav remix 1/2")
   if plot==True:
      plt.plot(X_sep[:,0])
      plt.plot(X_sep[:,1])
      plt.title('The unmixed channels')
      plt.show()
      
   return processingtime, X_sep

if __name__ == '__main__':

   #Signal to separate:
   #mixfile= "mix16000cubenoise.wav"
   mixfile = "mix16000.wav"
   processingtime, X_sep=separation_trinicon(mixfile, plot=True)
   
   samplerate, X = wav.read(mixfile)
   print("samplerate=", samplerate)
   chanout=2
   X_sep=X_sep*1.0/np.max(abs(X_sep))
   for c in range(chanout):
      os.system('espeak -s 120 "Separated Channel'+str(c)+' " ')
      playsound(X_sep[:,c]*2**15, samplerate, 1)
   
   
