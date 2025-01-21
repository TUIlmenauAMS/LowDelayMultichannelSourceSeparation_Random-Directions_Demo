#Program to provide a live sound card frontend for ICAabskl_puredelay_online, 
#by giving it a different "main" section.
#Including live audio output for headphones
#Using the callback function for pyaudio to avoid flutter in playback.
#Gerald Schuller, April 2019

import ICAabskl_puredelay_online2

PLAY = True
PassThrough = True
Stereo=True
Channel0=True
maxdelay = 50 #maximum expected delay, to fill up coeff array to constant length

def key_function(event):
    global PLAY
    global PassThrough
    # print('You pressed ' + event.char)
    if event.char == 'q':
      print('I am stopping')
      PLAY = False
    if event.char == 'p':
       PassThrough=True
    if event.char == 's':
       PassThrough=False
       
def togglepass():
   #toggles pass through on and off
   global PassThrough
   PassThrough=not(PassThrough)
   return
   
def quitbutton():
   #quits the program
   global PLAY
   PLAY=False
   return
   
def stereobutton():
   #switches between stereo and one channel
   global Stereo
   Stereo=True
   return
   
def channelbutton():
   #switches between channel 0 and 1
   global Channel0, Stereo
   Channel0=not(Channel0)
   Stereo=False
   return

"""
class sepoptim:
   def __init__(self, M=4, Blocksize):
   self.sigmemory=np.zeros((Blocksize*M,2))  #Internal signal memory, stereo
   #Initialize the filter memory:
   self.state0=np.zeros((maxdelay+1,M+1)) #fractional delay filter states for the length of the signal memory (M Blocks)
   #state1=np.zeros(maxdelay+1)
   self.state1=np.zeros((maxdelay+1,M+1)) 
"""

def callbackseparation(in_data, frame_count, time_info, flag):
   #global fulldata #global variables 
   global nchan, coeffs, state0, state1, PassThrough, Stereo, Channel0
   #print ("frame_count=", frame_count)
   #Xblock = np.fromstring(in_data, dtype=np.float32)
   #Source separation:
   #print("Block size:", int(frame_count))
   #samples = stream.read(Blocksize)
   Xblock = (struct.unpack( 'h' * nchan*frame_count, in_data));
   Xblock=np.array(Xblock)/32768.0/2.0
   Xblock=np.reshape(Xblock,(-1,nchan))
   #coeffs=[0.9, 0.9, 8.2, 8.2]
   #"""
   sigmemory[:-Blocksize,:]=sigmemory[Blocksize:,:] #shift old samples left
   sigmemory[-Blocksize:,:]=Xblock  #Write new block on right end
      
   #Xunm, coeffs, state0, state1 = blockseparationoptimization(coeffs, Xblock, state0, state1)
   state0[:,:-1]=state0[:,1:] #shift states 1 left to make space for the newest state
   state1[:,:-1]=state1[:,1:]
   Xunm, coeffs, state0[:,-1], state1[:,-1] = ICAabskl_puredelay_online2.blockseparationoptimization(coeffs, sigmemory, state0[:,-M-1], state1[:,-M-1])
   Xunm=Xunm[-Blocksize:,:]
   coeffs[2:]=np.clip(coeffs[2:],0,maxdelay)  #limit the delays to maxdelay
   #"""
   #Xunm, coeffs, state0, state1 = ICAabskl_puredelay_online2.blockseparationoptimization(coeffs, Xblock, state0, state1)
   #print("coeffs=", coeffs)
   #Store unmixed block:
   #X_del[m*Blocksize+np.arange(Blocksize),:]=Xunm
   #play out samples:
  
   print("PassThrough=", PassThrough)
   if PassThrough==False:  #press separated if key "s" is pressed:
      audio_data=Xunm
      #print("Separated")
   else:   #play input directly if key "p" is pressed
      audio_data=Xblock
      #print("Pass through")
   if not(Stereo):
      if Channel0:
         #mono playback:
         #audio_data=np.dot(audio_data, np.array([[1,1],[0,0]])) #duplicate channel 0 on both channels
         #spatial playback:
         audio_data=np.dot(audio_data, np.array([[1,0],[0,0.1]])) #attenuate ch.1
         print("Channel 0")
      else:
         #mono playback:
         #audio_data=np.dot(audio_data, np.array([[0,0],[1,1]]))  #duplicate channel 1 on both channels
         #spatial playback:
         audio_data=np.dot(audio_data, np.array([[0.1,0],[0,1]])) #attenuate ch.0
         print("Channel 1")
      #spatial playback, with negative attenuation factors to mot cancel but playing back the corresponind delay:
      print("coeffs=", np.hstack((-coeffs[0:2],coeffs[2:4])))
      audio_data, state00plb, state10plb =ICAabskl_puredelay_online2.unmixing(np.hstack((-coeffs[0:2],coeffs[2:4])), audio_data, state0plb, state1plb)
   #audio_data=np.reshape(audio_data,(-1,1))
   audio=np.clip(audio_data*2**15, -32000,32000)
   sound = audio.astype(np.int16).tobytes()
   #sound = audio_data.astype(np.float32).tostring()
   # fulldata = np.append(fulldata,audio_data) #saves filtered data in an array
   return (sound, pyaudio.paContinue)
   
#callbackseparation.sigmemory=np.zeros((Blocksize*M,2))
   
#run it from sound card:
if __name__ == '__main__':
   import scipy.io.wavfile as wav
   import scipy.signal 
   import os
   import matplotlib.pyplot as plt
   import pyaudio
   import numpy as np
   import struct
   import sys

   if sys.version_info[0] < 3:
       # for Python 2
       import Tkinter as Tk
   else:
       # for Python 3
       import tkinter as Tk   
       
   top = Tk.Tk()
   top.bind("<Key>", key_function)
   top.title("Separation Demo")
   #Tk.Label(text="Test", fg="black", bg="white")
   T = Tk.Text(top, height=6, width=30)
   #top.Text(top, height=4, width=30)
   #top.pack()
   #top.insert(Tk.END, "Passthrough:'p'\nSeparating: 's'\nEnd: 'q'\n")
   b = Tk.Button(top, text="Pass Through on/off", command= togglepass)
   bq = Tk.Button(top, text="Quit", command= quitbutton)
   bstereo = Tk.Button(top, text="Stereo", command= stereobutton)
   bchannel = Tk.Button(top, text="Toggle Channels", command= channelbutton)
   T.pack()
   T.insert(Tk.END, "Passthrough:'p'\nSeparating: 's'\nEnd: 'q'\n")
   b.pack()
   bstereo.pack()
   bchannel.pack()
   bq.pack()
   
   
   bytes = 2 #2 bytes per sample
   nchan = 2 #2
   fs = 32000  #Sampling Rate in Hz
   Blocksize=1024
   #sep=sepoptim(M=4, Blocksize=Blocksize)  #set up separation optimization object with memory size = 4 times Blocksize
   
   #Open sound card
   p = pyaudio.PyAudio()
   stream = p.open(format=p.get_format_from_width(bytes),
                channels=nchan,
                rate=fs,
                input=True,
                output=True,
                #input_device_index=3,
                frames_per_buffer=Blocksize, #frame: a stereo pair, 2 samples
                stream_callback=callbackseparation  #sep.
                )
   #Blocks=400 #Number of blocks to be read from the sound card
   #print("Blocks=", Blocks)
   
   #Initialize the filter memory:
   """
   state0=np.zeros(maxdelay+1)
   state1=np.zeros(maxdelay+1)
   """
   #"""
   M=4 #number of blocks for sliding window for optimization
   sigmemory=np.zeros((Blocksize*M,2))  #Internal signal memory, stereo
   #Initialize the filter memory:
   state0=np.zeros((maxdelay+1,M+1)) #fractional delay filter states for the length of the signal memory (M Blocks)
   state1=np.zeros((maxdelay+1,M+1)) 
   state0plb=np.zeros((maxdelay+1))
   state1plb=np.zeros((maxdelay+1))
   #"""
   
   #coeffs=[0.8, 0.8, 5.0, 5.0]
   coeffs=[1.0, 1.0, 1.0, 1.0]
   #coeffweights=np.array([0.1,0.1,1.0,1.0])*0.8
   #X_del=np.zeros((Blocks*Blocksize,2))
   #coeffweights=np.array([0.1,0.1,1.0,1.0])*0.8
   #Loop over the signal blocks:
   while PLAY:   #stops if key q is pressed
      T.delete(1.0, Tk.END)
      T.insert(Tk.END, "Keyboard Control:\n")
      T.insert(Tk.END, "Pass through:'p'\nSeparating: 's'\nEnd: 'q'\n")
      if PassThrough==True:
         T.insert(Tk.END, "Passing through\n")
      else:
         T.insert(Tk.END, "Separating\n")
      if Stereo:
         T.insert(Tk.END, "Stereo\n")
      else:
         if Channel0:
            T.insert(Tk.END, "Chanel 0\n")
         else:
            T.insert(Tk.END, "Chanel 1\n")
      top.update()
      
      stream.start_stream()
      
      
      #samples=samples.astype(int) #stereo signal
      #print("samples=", samples)
      #converting from short integers to a stream of bytes in "data":
      #data=struct.pack('h' * len(samples), *samples);
      #Writing data back to audio output stream: 
      #stream.write(data, CHUNK)
      
   stream.stop_stream()
   stream.close()
   print("Done")
   p.terminate()
      
   
