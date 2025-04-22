"""
Simple program to mix different sources in a simulated room using pyroomacoustics.
Gerald Schuller, 2021-08-09
"""
import pyroomacoustics as pra
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

# 16000 Hz sampling rate:
# """
# fs , audio0 = wavfile.read('sc03_16m.wav')
# fs , audio0 = wavfile.read('Schmoo16000.wav')
# fs , audio0 = wavfile.read('oscili_test_16.wav')
# fs , audio0 = wavfile.read('oscili_test_long.wav')
# fs , audio0 = wavfile.read('oscili_test_nonharmonic.wav')
# fs , audio0 = wavfile.read('Schmoo16000m.wav')
# fs , audio0 = wavfile.read('fantasy-orchestra_m16.wav')
# fs , audio0 = wavfile.read('sc03_16.wav')
# fs , audio0 = wavfile.read('pinkish16.wav')
# fs , audio0 = wavfile.read('oscili_test_ampl.wav')
# audio0 =1.0*audio0 #reduce volume
# audio0 =0.25*audio0[:,0] #reduce volume
# fs , audio1 = wavfile.read('SI889.wav')
# fs , audio1 = wavfile.read('espeakwav_16.wav')
# fs , audio1 = wavfile.read('espeaklong_16.wav')
# """
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


def room_mix(files, micsetup='stereo', plot=False, rt60=0.2):
   # Files: List of files with the audio sources
   # plot=True: Room setup is plotted
   # micsetup='cube' #'square' 'stereo' : possible microphone setups
   # writes the room mix to mix16000.wav

   fs, audio0 = wavfile.read(files[0])
   fs, audio1 = wavfile.read(files[1])

   # The desired reverberation time and dimensions of the room
   # rt60 = 0.1# seconds
   
   room_dim = [5, 4, 3]  # meters
   """
   MI: Comments:
   The calculation of material properties based on RT60 in Pyroomacoustics is an approximation and may not be entirely realistic when applied
   to real-time scenarios, especially in a stereo setup. RT60 provides an estimate of the overall reverberation characteristics of a room,
   but it does not account for dynamic changes, directional reflections, or spatial variations in absorption and diffusion that occur in real-time audio processing.
   As a result, while RT60-based calculations are useful for general acoustic modeling, they may not fully capture the complexity of real-world stereo configurations
   where precise spatial characteristics and directivity play a significant role.
   """

   # We invert Sabine's formula to obtain the parameters for the ISM simulator
   # e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
   # m = pra.Material(energy_absorption="hard_surface") # OR
   
   """
   MI: Comments:
   Here is an example of using a realistic room for RIR synthesis while incorporating e_absorption with real-world materials
   """

   m = pra.make_materials(
      ceiling="hard_surface",
      floor="carpet_cotton",
      east="plasterboard",
      west="plasterboard",
      north="wood_16mm",
      south="wood_16mm",
      )
   
   #print("e_absorption=", e_absorption, "max_order =", max_order)
   # Create the room
   # In main code ISM is use to calculate the RIR. The order is very high. Reason? I recommend order up to 2 or 3.
   """
   room = pra.ShoeBox(
       room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
   )
   """
   max_order = 2
   room = pra.ShoeBox(
      room_dim,
      fs=fs,
      #materials=pra.Material(e_absorption),
      materials=m,
      max_order=max_order,
      # ray_tracing=True,
      air_absorption=True
      )
   
   print("....", "\n", "....", "\n",
         "RT60 =  ", rt60, "\n"
         "ISM Order =  ", max_order, "\n"
         "Ceiling Material Coeffs =  ", m["ceiling"].absorption_coeffs, "\n"
         "Floor Material Coeffs =  ", m["floor"].absorption_coeffs, "\n"
         "East Wall Material Coeffs =  ", m["east"].absorption_coeffs, "\n"
         "West Wall Material Coeffs =  ", m["west"].absorption_coeffs, "\n"
         "North Wall Material Coeffs =  ", m["north"].absorption_coeffs, "\n"
         "South Wall Material Coeffs =  ", m["south"].absorption_coeffs, "\n"
         " .... " )

   # import a mono wavfile as the source signal
   # the sampling frequency should match that of the room

   # place the source in the room
   """
   MI: Comments:
   Source and reciever heights should be generally at the same level
   """
   room.add_source([2.5, 1.5, 1.2], signal=audio1, delay=0.0)
   room.add_source([2.5, 3.3, 1.2], signal=audio0, delay=0.0)

   # define the locations of the microphones

# COMMENT: WHY SOURCE AND RECEIVER ARE AT DIFFERENTHEIGHTS?

# Stereo microphones:
   if micsetup == 'stereo':
      mic_locs = np.c_[
         [3.0, 2.0, 1.2],  # mic 1
         [3.0, 2.2, 1.2],  # mic 2 , 20cm in y direction from the first mic
         ]

   # Quadratic microphone locations, flat:
   if micsetup == 'square':
      mic_locs = np.c_[
         [3.0, 2.0, 1.2],  # mic 1
         [3.0, 2.2, 1.2],  # mic 2 , 20cm in y direction from the first mic
         [3.2, 2.0, 1.2],  # mic 3 , 20cm in x direction behind the first pair
         [3.2, 2.2, 1.2],  # mic 4
         ]
      
   # Cube microphone locations:
   if micsetup == 'cube':
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

   if plot == True:
        fig, ax = room.plot()
        ax.set_xlim([0, 6])
        ax.set_ylim([0, 5])
        ax.set_zlim([0, 3])
        plt.show()

    # Run the simulation (this will also build the RIR automatically)
   room.simulate()
   
   room.mic_array.to_wav(
        f"mix16000.wav",
        norm=True,
        bitdepth=np.int16,
    )
   print("wrote to mix16000.wav")
   if plot==True:
      room.plot_rir()
      plt.show()
  
   return


if __name__ == "__main__":
   # files=('pinkish16.wav', 'espeakwav_16.wav')
   files = ('espeakfemale_16.wav', 'espeakwav_16.wav')
   # room_mix(files, micsetup='cube', plot=True)
   room_mix(files, micsetup='stereo', plot=True, rt60=0.4)
   
   
   """
   MI: Comments:
      Findings:
         From here we are getting a sound (i.e. music etc.) wav file as a results of two speakers recorded at the a stereo steup of microphones (That is all)
   """


