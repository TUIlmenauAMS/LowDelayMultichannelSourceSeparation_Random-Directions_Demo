"""
Simple program to mix different sources in a simulated room using pyroomacoustics.
Gerald Schuller, 2021-08-09
"""
import pyroomacoustics as pra
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import fft, ifft, fftshift

#16000 Hz sampling rate:
#"""
#fs , audio0 = wavfile.read('sc03_16m.wav')
#fs , audio0 = wavfile.read('Schmoo16000.wav')
#fs , audio0 = wavfile.read('oscili_test_16.wav')
#fs , audio0 = wavfile.read('oscili_test_long.wav')
#fs , audio0 = wavfile.read('oscili_test_nonharmonic.wav')
#fs , audio0 = wavfile.read('Schmoo16000m.wav')
#fs , audio0 = wavfile.read('fantasy-orchestra_m16.wav')
#fs , audio0 = wavfile.read('sc03_16.wav')
#fs , audio0 = wavfile.read('pinkish16.wav')
#fs , audio0 = wavfile.read('oscili_test_ampl.wav')
#audio0 =1.0*audio0 #reduce volume
#audio0 =0.25*audio0[:,0] #reduce volume
#fs , audio1 = wavfile.read('SI889.wav')
#fs , audio1 = wavfile.read('espeakwav_16.wav')
#fs , audio1 = wavfile.read('espeaklong_16.wav')
#"""
"""
#44100 Hz sampling rate:
fs , audio0 = wavfile.read('fantasy-orchestra.wav')
print("audio0.shape=", audio0.shape)
audio0=0.25*audio0[:,0] #make mono
print("audio0.shape=", audio0.shape)
fs , audio1 = wavfile.read('SI889_44100.wav')
"""
"""
#48000 Hz sampling rate:
fs , audio0 = wavfile.read('/home/schuller/Programs/PythonProgs/Schmoo.wav')
print("audio0.shape=", audio0.shape)
audio0=0.25*audio0[:,0] #make mono
print("audio0.shape=", audio0.shape)
fs , audio1 = wavfile.read('SI889_48000.wav')
"""

def room_mix(files, micsetup='stereo', plot=True, rt60=0.1):
   #Files: List of files with the audio sources
   #plot=True: Room setup is plotted
   #micsetup='cube' #'square' 'stereo' : possible microphone setups
   #writes the room mix to mix16000.wav

   fs , audio0 = wavfile.read(files[0])
   fs , audio1 = wavfile.read(files[1])

   # The desired reverberation time and dimensions of the room
   #rt60 = 0.1# seconds
   print("rt60=", rt60)
   
   room_dim = [5, 4, 2.5]  # meters

   # We invert Sabine's formula to obtain the parameters for the ISM simulator
   e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
   
   print("e_absorption=", e_absorption,"max_order =", max_order) 

   # Create the room
   room = pra.ShoeBox(
       room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
   )

   # import a mono wavfile as the source signal
   # the sampling frequency should match that of the room

   # place the source in the room
   room.add_source([2.5, 1.5, 1.50], signal=audio1, delay=0.0)
   room.add_source([2.5, 3.3, 1.50], signal=audio0, delay=0.0)

   # define the locations of the microphones


   #Stereo microphones:
   if micsetup=='stereo':
      mic_locs = np.c_[
          [3.0, 2.0, 1.2],  # mic 1
          [3.0, 2.2, 1.2],  # mic 2 , 20cm in y direction from the first mic
      ]


   #Quadratic microphone locations, flat:
   if micsetup=='square':
      mic_locs = np.c_[
          [3.0, 2.0, 1.2],  # mic 1
          [3.0, 2.2, 1.2],  # mic 2 , 20cm in y direction from the first mic
          [3.2, 2.0, 1.2],  # mic 3 , 20cm in x direction behind the first pair
          [3.2, 2.2, 1.2],  # mic 4
      ]


   #Cube microphone locations:
   if micsetup=='cube':
      mic_locs = np.c_[
          [3.0, 2.0, 1.2],  # mic 1
          [3.0, 2.2, 1.2],  # mic 2 , 20cm in y direction from the first mic
          [3.2, 2.0, 1.2],  # mic 3 , 20cm in x direction behind the first pair
          [3.2, 2.2, 1.2],  # mic 4
          [3.0, 2.0, 1.4],  # mic 5 , 20cm in z direction above first square
          [3.0, 2.2, 1.4],  # mic 6 , 20cm in y direction from the first mic
          [3.2, 2.0, 1.4],  # mic 7
          [3.2, 2.2, 1.4],  # mic 8
      ]

   # finally place the array in the room
   room.add_microphone_array(mic_locs)

   if plot==True:
      fig, ax = room.plot()
      ax.set_xlim([0, 6])
      ax.set_ylim([0, 5])
      ax.set_zlim([0, 3]);
      plt.show()
   
   room.rir_duration = 0.5 # in seconds
   
   # Run the simulation (this will also build the RIR automatically)
   room.simulate()

   room.mic_array.to_wav(
       f"mix16000.wav",
       norm=True,
       bitdepth=np.int16,
   )
   print("wrote to mix16000.wav")
   print("room.rir[0][0].shape=", room.rir[0][0].shape)
   #"""
   if plot== True:
      room.plot_rir()  #Index order: mic, source
      Lrir=min(len(room.rir[0][1]), len(room.rir[0][0]))
      print("Lrir=", Lrir)
      #Relative Room Impulse Response for mic 0 from source 1
      rrir0=np.real(ifft(fft(room.rir[0][1][:Lrir]) /fft(room.rir[0][0][:Lrir]) ))
      #Relative Room Impulse Response for mic 1 from source 0
      rrir1=np.real(ifft(fft(room.rir[1][0][:Lrir]) /fft(room.rir[1][1][:Lrir]) ))

      
      plt.figure()
      plt.plot(rrir0)
      plt.title("Relative Room Impulse Response for mic 0 from source 1")
      plt.show()
      maxind0=np.argmax(rrir0)
      print("Attenuation0", rrir0[maxind0])
      if maxind0> Lrir/2:
          maxind0=maxind0-Lrir
      print("Delay0=", maxind0)
      
      plt.figure()
      plt.plot(rrir1)
      plt.title("Relative Room Impulse Response for mic 1 from source 0")
      plt.show()
      maxind1=np.argmax(rrir1)
      print("Attenuation1", rrir1[maxind1])
      if maxind1> Lrir/2:
          maxind1=maxind1-Lrir
      print("Delay1=", maxind1)
      
   #"""
      
   return

if __name__ == "__main__":
   files=('pinkish16.wav', 'espeakwav_16.wav')
   #files=('espeakfemale_16.wav', 'espeakwav_16.wav')
   #room_mix(files, micsetup='cube', plot=True)
   room_mix(files, micsetup='stereo', plot=True)


