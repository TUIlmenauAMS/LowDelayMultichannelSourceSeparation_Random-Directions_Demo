#Multi-Channel source separation (for instance 7 channels) using fractional delays and attenuations between the 
#microphone signals as relative impulse response, with offline update as optimization.
#Version 2, Output: 2 channels. Needs rectangular unmixing matrices,
#with a Laplacian for the equality of input power and output power.
#Using the optimization of random directions in an online fasshion, in parallel in the beginning
#Gerald Schuller, July 2020

import numpy as np
import scipy.signal
import scipy.stats as stats

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

def allp_delayfilt(tau):
    '''
    produces a Fractional-delay All-pass Filter
    Arguments:tau = fractional delay in samples. When 'tau' is a float - sinc function. When 'tau' is an integer - just impulse.
    type of tau: float or int
    :return:
        a: Denumerator of the transfer function
        b: Numerator of the transfer function
    '''
    #L = max(1,int(tau)+1) with the +1 the max doesn't make sense anymore
    L = int(tau)+1
    n = np.arange(0,L)
    # print("n", n)

    a_0 = np.array([1.0])
    a = np.array(np.cumprod( np.divide(np.multiply((L - n), (L - n - tau)) , (np.multiply((n + 1), (n + 1 + tau))) ) ))
    a = np.append(a_0, a)   # Denumerator of the transfer function
    # print("Denumerator of the transfer function a:", a)

    b = np.flipud(a)     # Numerator of the transfer function
    # print("Numerator of the transfer function b:", b)

    return a, b
    
def abskl_multichan(Xunm, chanout):  
   #computes the normalized magnitude of the (unmixed) channels Xunm and then applies 
   #the Kullback-Leibler divergence, and returns its negative value for minimization
   chanout=Xunm.shape[1]
   X_abs=np.abs(Xunm)
   #normalize to sum()=1, to make it look like a probability:
   for tochan in range(chanout):
      X_abs[:,tochan]=X_abs[:,tochan]/(np.sum(X_abs[:,tochan])+1e-6)
   #print("coeffs=", coeffs)
   #Kullback-Leibler Divergence:
   #print("KL Divergence calculation")
   abskl=0.0
   for k0 in range(chanout-1):
     for k1 in range(k0+1,chanout):
        abskl+= np.sum( X_abs[:,k0] * np.log((X_abs[:,k0]+1e-6)/(X_abs[:,k1]+1e-6)) )
        abskl+= np.sum( X_abs[:,k1] * np.log((X_abs[:,k1]+1e-6)/(X_abs[:,k0]+1e-6)) ) #for symmetry
   return -abskl
   
def entropysimilarity(X, k0, k1):
    #Computes the KL Divergence between channels k0 and k1
    hist0, bins =np.histogram(X[:,k0],bins=10000)
    hist1, bins =np.histogram(X[:,k1],bins=10000)
    similarity= -stats.entropy(hist0+1e-6,hist1+1e-6) #KL Divergence
    #print("similarity= ", similarity)
    return similarity
    
def klhist_multichan(Xunm, chanout):  
   #computes the normalized magnitude of the (unmixed) channels Xunm and then applies 
   #the Kullback-Leibler divergence, and returns its negative value for minimization
   chanout=Xunm.shape[1]
   
   #normalize to sum()=1, to make it look like a probability:
   #for tochan in range(chanout):
   #   X_abs[:,tochan]=X_abs[:,tochan]/(np.sum(X_abs[:,tochan])+1e-6)
   #print("coeffs=", coeffs)
   #Kullback-Leibler Divergence:
   #print("KL Divergence calculation")
   abskl=0.0
   for k0 in range(chanout-1):
     for k1 in range(k0+1,chanout):
        abskl+= entropysimilarity(Xunm,k0,k1)
        abskl+= entropysimilarity(Xunm,k1,k0) #make it symmetric
   return -abskl
   
def unmixing(coeffs, X, chanout, maxdelay=20, state=[]):
   #Applies an anmixing matrix build from coeffs to a multi-channel signal X
   #Each column is a channel
   #Arguments: 
   #Coeffs: 2d array which contains attenuations (from,to) in the first column (from: input signal, to: output signal),
   # and the delays, delay(from,to) in the second column.
   #X= multichannel signal from the microphones, each column is a channel
   #maxdelay: maximum delay for varying delays (default:20), depends on the setup (distance of microphones, sampling rate)
   #should be larger than: max distance of microphones (in meter) / (300m/s) * Sampling rate (Hz)
   #but not too large because it increases computational complexity.
   #chanout: number of signals to separate in the output
   #state = states from previous delay filter run, each column is a filter set
   #3-dim array, a filter set is: state[from,to,:]
   #
   #Returns the resulting (unmixed) stereo signal X_sep
   #Unmixing Process:
   #Delayed and attenuated versions of opposing microphones are subtracted:
   #Xsep[:,0]= att[0,0] * delay[0,0](X0)- att[1,0] * delay[1,0](X1) - att[2,0] * delay[2,0](X2)...
   #...

   chanin=X.shape[1] #number of channels of the input signal
   #chanout=2;
   #make the attenuation matrix from the coefficients, with 1's on the diagonal:
   #att=coeff2matrix(coeffs[:,0])
   att=np.reshape(coeffs[:,0],(chanin,chanout))
   
   #att[0,:]=[1,1]
   #print("att=", att)
   #make the delay matrix, with 1's on the diagonal (delay in samples):
   #delay=coeff2matrix(coeffs[:,1])
   delay=np.reshape(coeffs[:,1],(chanin,chanout))
   delay=abs(delay)
   #print("delay=", delay)
   #delay-=np.eye(delay.shape[0]) #make the delays on the diagnal zero 
   #(signal to itself). Without this, performance is reduced.
   #print("coeffs=", coeffs, ", att=", att, ", delay=", delay)
   
   #chanout=chanin
   #print("Channels=", chanin) 
   X_sep=np.zeros((X.shape[0],chanout)) #Shape of the "chanout" channel output signal

   #maxdelay = maximum expected delay, to fill up coeff array to constant length
   #print("unmixing state.shape=", state.shape)
   if state!= []:
      maxdelay=state.shape[-1]-1  #length of the filter states (last dimension) -1
   #print("maxdelay=", maxdelay)
   #Attenuations:
   att=np.clip(att, -1.5, 1.5) #limit possible attenuations
   #delay=np.abs(coeffs[2:4]) #allow only positive values for delay, for CG optimizer
   delay=np.clip(delay,0,maxdelay) #allow only range of 0 to maxdelay
   #print("delay=", delay)
   #delay filters for fractional delays:
   a=np.zeros((chanin, chanout, maxdelay+2))
   b=np.zeros((chanin, chanout, maxdelay+2))
   for fromchan in range(chanin):
      for tochan in range(chanout):  #2 channel output
         a0,b0=allp_delayfilt(delay[fromchan, tochan])
         a[fromchan,tochan,:]= np.append(a0,np.zeros(maxdelay+2-len(a0)))
         b[fromchan,tochan,:]= np.append(b0,np.zeros(maxdelay+2-len(b0)))
         
   #print("a.shape=", a.shape, "b.shape=", b.shape,"maxdelay=", maxdelay)
   #a0=np.append(a0,np.zeros(maxdelay+2-len(a0)))
   #b0=np.append(b0,np.zeros(maxdelay+2-len(b0)))
   #print("Len of a0, b0:", len(a0), len(b0))
   
   #The channels are delayed with the allpass filter:
   Y=np.zeros((X.shape[0], X.shape[1], chanout)) #The delayed signals, shape: signal length, input chan, output chan
   #print("X.shape=", X.shape,"Y.shape=", Y.shape)
   if state!=[]:
      for fromchan in range(chanin):
         for tochan in range(chanout):  #2 channel output
            Y[:,fromchan, tochan], state[fromchan, tochan,:]=scipy.signal.lfilter(b[fromchan,tochan,:], a[fromchan,tochan,:],X[:,fromchan], zi=state[fromchan, tochan,:])
   else:
      for fromchan in range(chanin):
         for tochan in range(chanout):  #2 channel output
            Y[:,fromchan, tochan] =scipy.signal.lfilter(b[fromchan,tochan,:], a[fromchan,tochan,:],X[:,fromchan])
         
   #Delayed and attenuated versions of opposing microphones are subtracted:
   for tochan in range(chanout):
      X_sep[:,tochan]=np.dot(Y[:,:,tochan], att[:,tochan])
   
   #X_sep[:,0]=X[:,0]-a[0]*y1
   #X_sep[:,1]=X[:,1]-a[1]*y0
   return X_sep #, state
   
def wrapperfunc(coeffs1d, X, chanout):
   #wrapper function for scipy.minimize
   #chanin=int(len(coeffs1d)/chanout/2)
   #print("chanin=2", chanin, "coeffs1d=", coeffs1d)
   coeffs=np.reshape(coeffs1d,(-1,2))
   args=(X,chanout)
   err=objfunc(coeffs, args)
   print("err=", err)
   return err
   
def objfunc(coeffs, args):
#def objfunc(coeffs, X, chanout):
   #error function for optimimzation
   #arguments:
   #coeffs : coefficients to be optimized
   #X: The multichannel audio signal
   #chanout: the number of sources to be separated
   #returns: the error value
   
   #print("args=", args)
   (X, chanout)=args
   chanin=X.shape[1] #number of channels of the input signal
   #print("chanout=", chanout, "chanin=", chanin, "coeffs.shape=", coeffs.shape)
   Xunm1  =unmixing(coeffs, X, chanout)
   #average input power:
   inpow=np.sum(X**2)/X.shape[1]
   #sum of power of all sources:
   outpow=np.sum(Xunm1**2) #/chanout
   
   #include Laplacian for preserving power 
   #(1 input channel should have the energy of all separated sources):
   #negabskl=abskl(Xunm1, chanout) +0.1*abs(outpow/inpow-1)
   #"""
   #power preservation for each output channel, sum over the input channels:
   inpow=sum(sum(X**2))/X.shape[1]
   #print("inpow=", inpow)
   outpow=np.sum(Xunm1**2, axis=0)
   #print("outpow=", outpow)
   negabskl=abskl_multichan(Xunm1, chanout) +0.1*sum(abs(outpow/(inpow)-1))
   #negabskl=klhist_multichan(Xunm1, chanout) +0.1*sum(abs(outpow/(inpow)-1))
   #"""
   """
   #using the sum of squares of the attenuation coefficients for power preservation (assuming gauss distributed noise like signal):
   att=np.reshape(coeffs[:,0],(chanin,chanout))
   attsq=np.zeros(chanout)
   #sum of squares of the attenuation coefficients for each output channel:
   for chan in range(chanout):
      attsq[chan]=sum(att[:,chan]**2)
   #side condition: make the sum of squares of the attenuation coefficients as close as possible to 1 for power preservation:
   negabskl=abskl(Xunm1, chanout) +0.1*sum(abs(attsq-1))
   """
   """
   #Spectral unflatness as side condition:
   unflatness=freqresp(coeffs, chanin, chanout)
   negabskl=abskl(Xunm1, chanout) +0.01*unflatness
   """
   return negabskl
   
def freqresp(coeffs, chanin, chanout):
   #estimating the frequency response of a signal in front (no delays between mics):
   
   att=np.reshape(coeffs[:,0],(chanin,chanout))
   #print("att=", att)
   delay=np.reshape(coeffs[:,1],(chanin,chanout))
   #print("delay=", delay)
   delay=abs(delay)
   #print("delay=", delay, "max(delay)=", np.max(delay))
   maxdelay=int(round(np.max(delay)))
   #print("maxdelay=", maxdelay)
   #creating the impulse responses to the output channel as sum of the impulse responses from the input channels:
   impresp=np.zeros((chanout, maxdelay+1))
   unflatness=0.0
   for chan in range(chanout):
      #print("int(np.round(delay[:,chan]))=", (np.round(delay[:,chan])).astype(int))
      impresp[chan, (np.round(delay[:,chan])).astype(int)] += att[:, chan]
      w,H=scipy.signal.freqz(impresp[chan,:], worN=64,)
      H=20*np.log10(abs(H)+1e-5)
      ave=np.mean(abs(H))*np.ones(len(H))
      #plt.plot(w,ave)
      unflatness=+np.mean(abs(abs(H)-ave))
      #plt.plot(w,abs(H))
   """
   print("unflatness=", unflatness)
   print("impresp=", impresp)
   plt.title("Magnitude Frequency Response of a source with equal delays to the mics")
   plt.show()
   """
   return unflatness

def separation_randdir(mixfile, plot=True):
#Separates 2 audio sources from the multichannel mix in the mixfile,
#Using sum and delay unmixing with random direction optimization
#plot=True plots the resulting unmixed wave forms.

   import scipy.io.wavfile as wav
   import scipy.optimize as opt
   import os
   import time
   import matplotlib.pyplot as plt
   #import optimrandomdir_linesearch as optimrandomdir
   import optimrandomdir_parallel_linesearch as optimrandomdir #parallel seems to be better, lower f.
   #import optimrandomdir_parallel_linesearch_adaptstep as optimrandomdir
   #import optimrandomdir_parallel as optimrandomdir #parallel seems to be better, lower f.
   #import optimrandomdir_BGDadapt as optimrandomdir
   #import GLD_fast
   
   samplerate, X = wav.read(mixfile)
   print("X.shape=", X.shape)
   X=X*1.0/np.max(abs(X)) #normalize
   
   N=X.shape[1] #number of microphones or signals to separate
   print("Number of channels N:", N)
   chanout=2 #2 output channels
   
   coeffs=np.random.random((N*chanout,2))  #column 0: attenuation, column 1: delays
   #coeffs=np.ones((N*chanout,2)); coeffs[:,1]=0;  #column 0: attenuation, column 1: delays
   print("coeffs=", coeffs)
   coeffdeviation=1.0
   #coeffdeviation=np.ones(coeffs.shape)
   #coeffdeviation[:,0]*=0.1 #attenuations: less deviation
   #print("coeffdeviation=", coeffdeviation)
   #err=objfunc(coeffs, (X, chanout))
   #err=objfunc(coeffs, X, chanout)
   #print("err=", err)
   
   #siglen=X.shape[0] #length of signal
   siglen=max(X.shape)
   print("siglen=", siglen)
   #optimization:
   #Accumulating blocks of the beginning of the signal for the optimization
   #This can be done in parallel in the beginning
   #for low delay processing with only a few samples delay.
   #Xblock=X[0:50000,:] #short time piece of the signal
   blocksize=8000
   blocks= int(siglen/blocksize)
   print("blocks=", blocks)
   blockaccumulator=X[0:blocksize,:]
   blockno=0
   
   starttime=time.time()
   for ob in range(1): #sub-periods after which to run the optimization, outer blocks, seems best for 1 outer block.
      #Accumulate part of the signal in a signal "accumulator" of size "blocksize" (8000 samples, or 0.5s):
      for i in range(min(blocks,16)): #accumulate audio blocks over about 3 seconds:
         blockaccumulator=0.98*blockaccumulator + 0.02*X[blockno*blocksize+np.arange(blocksize)]
         blockno+=1
      #optimize the unmixing coefficients "coeffs":
      coeffs= optimrandomdir.optimrandomdir(objfunc, coeffs, args=(blockaccumulator,chanout), coeffdeviation=coeffdeviation, iterations=1000, startingscale=4.0, endscale=0.0)
   #coeffs= optimrandomdir.optimrandomdir(objfunc, coeffs, args=(X,chanout), coeffdeviation=coeffdeviation, iterations=400, startingscale=4.0, endscale=0.0) #startingscale=4 might be useful for finding the right delay, the endscale=0.1 for the right attenuations, and it stays flexible at that point.
   #coeffs = GLD_fast.gldfast(objfunc, coeffs, args=(X,chanout), iterations=150, startingscale=4)
   """
   coeffs1d = opt.minimize(wrapperfunc, coeffs , args=(X,chanout), method='CG',options={'disp':True, 'maxiter': 10})
   coeffs1d=coeffs1d.x
   print("coeffs1d=", coeffs1d)
   coeffs=np.reshape(coeffs1d,(-1,2))
   """
   endtime=time.time()
   processingtime=endtime-starttime
   print("Duration of optimization:", endtime-starttime, "sec.")
   print("coeffs=", coeffs)
   #Estimation of frequency response across unmixing:
   #chanin=X.shape[1] #number of channels of the input signal
   #unflatness=freqresp(coeffs, chanin, chanout)
   #print("unflatness=", unflatness)
   
   #unmixing from optimization solution for testing:
   #starttime=time.time()
   X_sep  =unmixing(coeffs, X, chanout)
   #endtime=time.time()

   wav.write("sepchan_randdir_online.wav",samplerate,np.int16(np.clip(X_sep*2**15,-2**15,2**15-1)))
   print("Written to sepchan_randdir_online.wav")
   if plot==True:
      plt.plot(X_sep[:,0])
      plt.plot(X_sep[:,1])
      plt.title('The unmixed channels')
      plt.show()
      
   return processingtime, X_sep
   

if __name__ == '__main__':
   import os
   import scipy.io.wavfile as wav

   #samplerate, X = wav.read("/home/schuller/Nextcloud/Teaching/GeraldsBSS/7_channel_test_audio5.wav")
   #X=X[:,:7] #last channel 7 is empty
   #samplerate, X = wav.read("/home/schuller/Nextcloud/Teaching/GeraldsBSS/7_channel_simulated_audio_RT60-0_05.wav")
   #samplerate, X = wav.read("/home/schuller/Nextcloud/Teaching/GeraldsBSS/7_channel_simulated_audio_RT60-0_1.wav")
   #samplerate, X = wav.read("/home/schuller/Nextcloud/Teaching/GeraldsBSS/cubic_RT60-01s_dist-1m_0.wav")
   #samplerate, X = wav.read("/home/schuller/Nextcloud/Teaching/GeraldsBSS/8_channels_mixed_signals_bad/test_audio_0.wav")
   #samplerate, X = wav.read("/home/schuller/Nextcloud/Teaching/GeraldsBSS/4_channel_test_mix.wav")
   #X=X[:,1::3] #X=X[:,3::3] #start at 1 much better than at 3!
   
   #X=X[:,2::2]
   #samplerate, X = wav.read("7_channel_test_audio.wav")
   #samplerate, X = wav.read("testfile_IDMT_dual_mic_array.wav")
   #samplerate, X = wav.read("2_channel_test_audio.wav")
   #samplerate, X = wav.read("3_channel_test_audio.wav")
   #samplerate, X = wav.read("4_channel_test_audio.wav")
   #samplerate, X = wav.read("stereomoving.wav")
   #samplerate, X = wav.read("stereo_record_14.wav")
   #samplerate, X = wav.read("stereovoices.wav")
   #samplerate, X = wav.read("stereovoicemusic4.wav")
   #samplerate, X = wav.read("Schmoo.wav")
   #samplerate, X = wav.read("s_chapman.wav")
   #samplerate, X = wav.read("rockyou_stereo.wav")
   #samplerate, X = wav.read("sepchanstereo.wav")
   #samplerate, X = wav.read("stereotest2.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix44100.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix48000.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000_short.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000_long.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000_nh.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000Schmoo.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000fantasyorchestra.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000sc03.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000noise.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000ampl.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000ampquadratictones.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000ampcubetones.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000cubetones.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000cubefantasy.wav")
   #samplerate, X = wav.read("/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000cubenoise.wav")
   
   mixfile="/home/schuller/Documents/ConfJournalsVortr3/2021-11Asilomar/SourcesSoftware/mix16000cubenoise.wav"
   #Unmixing:
   processingtime, X_sep= separation_randdir(mixfile, plot=True)
   
   print("Duration of unmixing:", processingtime, "sec.")
   X_sep=X_sep*1.0/np.max(abs(X_sep))
   chanout=X_sep.shape[1] #2 output channels
   samplerate, X = wav.read(mixfile)
   for c in range(chanout):
      os.system('espeak -s 120 "Separated Channel'+str(c)+' " ')
      playsound(X_sep[:,c]*2**15, samplerate, 1)
      
   """
   from mir_eval.separation import bss_eval_sources
   sdr, sir, sar, perm = bss_eval_sources(ref[:,:m], y[:,:m])
   """
   
