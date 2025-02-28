"""
Description:
This Python script performs blind source separation (BSS) of two audio sources recorded using a stereo microphone.
The separation is based on fractional delay and attenuation differences between the two microphone signals.
The algorithm is optimized for real-time applications with low delay and low computational complexity.
"""

# 2 Channel source separation using fractional delays and attenuations between the
# microphone signals as relative impulse response, and online update as optimization
# for real time applications with low delay and low complexity.
# Based on ICAabskl_puredelay_lowpass.py
# With only random directions, without +- for them.
# Gerald Schuller, April 2019

import numpy as np
import scipy.signal
maxdelay = 60  # maximum expected delay, to fill up coeff array to constant length


def allp_delayfilt(tau):
    '''
    Generates a fractional-delay all-pass filter to simulate small delays.
    Arguments:tau = fractional delay in samples. When 'tau' is a float - sinc function. When 'tau' is an integer - just impulse.
    type of tau: float or int
    Uses a recursive formula to compute the filter coefficients.
    Ensures phase response remains unchanged while delaying the signal.
    :return:
        a: Denumerator of the transfer function
        b: Numerator of the transfer function
    '''
    # L = max(1,int(tau)+1) with the +1 the max doesn't make sense anymore
    L = int(tau)+1
    n = np.arange(0, L)
    # print("n", n)

    a_0 = np.array([1.0])
    a = np.array(np.cumprod(np.divide(np.multiply(
        (L - n), (L - n - tau)), (np.multiply((n + 1), (n + 1 + tau))))))
    a = np.append(a_0, a)   # Denumerator of the transfer function
    # print("Denumerator of the transfer function a:", a)

    b = np.flipud(a)     # Numerator of the transfer function
    # print("Numerator of the transfer function b:", b)

    return a, b


def unmixing(coeffs, X, state0, state1):
    """
    Uses an adaptive filter to separate two mixed audio sources.
    The filter applies different attenuations and delays to the signals.
    Applies an anmixing matrix build from coeffs to a stereo signal X
    Arguments:
       coeffs = (attenuation 0, attenuation 1, delay0, delay1)
       X= Stereo signal from the microphones
       state0, state1 = states from previous delay filter run, important if applied
       to consecutive blocks of samples.
    Returns:
       The resulting (unmixed) stereo signal X_del
    Unmixing Process:
       The delayed and attenuated signals are subtracted to perform separation:
          Attenuation values are clipped to prevent excessive amplification.
          Delay values are adjusted within [0, maxdelay] to ensure stability.
          Fractional-delay filters a0, b0, a1, b1 are created
    Unmixing equations:
       Xdel0= X0- att0 * del0(X1)
       Xdel1= X1- att1 * del1(X0)

    """
    # print("coeffs =", coeffs)
    X_del = np.zeros(X.shape)
    # maxdelay = maximum expected delay, to fill up coeff array to constant length
    maxdelay = len(state0)-1
    # Attenuations:
    a = np.clip(coeffs[0:2], -1.5, 1.5)  # limit possible attenuations
    # delay=np.abs(coeffs[2:4]) #allow only positive values for delay, for CG optimizer
    # allow only range of 0 to maxdelay
    delay = np.clip(coeffs[2:4], 0, maxdelay)
    # print("delay=", delay)
    # delay filters for fractional delays:
    # using allpass delay filter:
    a0, b0 = allp_delayfilt(delay[0])
    a0 = np.append(a0, np.zeros(maxdelay+2-len(a0)))
    b0 = np.append(b0, np.zeros(maxdelay+2-len(b0)))
    a1, b1 = allp_delayfilt(delay[1])
    a1 = np.append(a1, np.zeros(maxdelay+2-len(a1)))
    b1 = np.append(b1, np.zeros(maxdelay+2-len(b1)))
    # print("Len of a0, b0:", len(a0), len(b0))
    # Both channels are delayed with the allpass filter:
    y1, state1 = scipy.signal.lfilter(b0, a0, X[:, 1], zi=state1)
    y0, state0 = scipy.signal.lfilter(b1, a1, X[:, 0], zi=state0)
    # Delayed and attenuated versions of opposing microphones are subtracted:
    X_del[:, 0] = X[:, 0]-a[0]*y1
    X_del[:, 1] = X[:, 1]-a[1]*y0
    return X_del, state0, state1


def pdf(x, bins):
    """
    Function to extimate the probability density distribution of a signal (in a 1d array)
    arguments: x: 1d array, contains a signal
    bins: # of bins for the histogram computation
    returns:  the pdf   

    """
    hist, binedges = np.histogram(x, bins)
    # print("hist=", hist)
    pdf = 1.0*hist/(np.sum(hist)+1e-6)
    return pdf


def kldivergence(P, Q):
    """
    Measures the difference between the PDFs of the two separated signals:
       Computes the normalized magnitude of the (unmixed) channels Xunm and then applies
       The Kullback-Leibler divergence, and returns its negative value for minimization
    Helps evaluate the separation quality

    Note: Function to compute the Kullback-Leiber-Divergence, see: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    Args: P,Q: pdf of 2 signals
    Return: L Divergence D_KL(P||Q)

    """
    abskl = np.sum(P * np.log((P+1e-6)/(Q+1e-6)))
    return abskl


def abskl(Xunm):
    """
    Computes absolute KL-divergence between the unmixed signals.
    Normalization: The absolute values of the separated signals are normalized to make them behave like probability distributions.
    Goal: Minimize KL-divergence to make each signal statistically independent.

    """
    X_abs = np.abs(Xunm)
    # normalize to sum()=1, to make it look like a probability:
    X_abs[:, 0] = X_abs[:, 0]/np.sum(X_abs[:, 0])
    X_abs[:, 1] = X_abs[:, 1]/np.sum(X_abs[:, 1])
    # print("coeffs=", coeffs)
    # Kullback-Leibler Divergence:
    # print("KL Divergence calculation")
    abskl = np.sum(X_abs[:, 0] * np.log((X_abs[:, 0]+1e-6)/(X_abs[:, 1]+1e-6)))
    # for symmetry
    abskl += np.sum(X_abs[:, 1] *
                    np.log((X_abs[:, 1]+1e-6)/(X_abs[:, 0]+1e-6)))
    return -abskl


def minabsklcoeffs(coeffs, Xblock, state0, state1, maxdelay):
    # computes the normalized magnitude of the channels and then applies
    # the Kullback-Leibler divergence
    Xunm0, state00, state10 = unmixing(coeffs, Xblock, state0, state1)
    negabskl0 = abskl(Xunm0)
    return negabskl0


def playsound(audio, samplingRate, channels):
    # funtion to play back an audio signal, in array "audio"
    import pyaudio
    p = pyaudio.PyAudio()
    # open audio stream

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplingRate,
                    output=True)

    audio = np.clip(audio, -2**15, 2**15-1)
    sound = (audio.astype(np.int16).tostring())
    # sound = (audio.astype(np.int16).tostring())ICAabskl_puredelay_lowpass.py
    stream.write(sound)

    # close stream and terminate audio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    return


def blockseparationoptimization(coeffs, Xblock, state0, state1):
    """
    This function optimizes the unmixing process using a random search approach:
       Starts with initial attenuation/delay values.
       Evaluates separation quality using KL-divergence.
          Reads in a block of a stereo signal, improves the unmixing coefficients, applies them and returns the unmixed block
       Applies small random variations to the coefficients.
       Keeps the new coefficients if they improve the separation.
       Updates the states of the filters to maintain continuity.
    Arguments:
       coeffs: array of the 4 unmixing coefficients
       Xblock: Block of the stereo signal to unmix
       state0, state1: Filter states for the IIR filter, length is maxdelay +1
    Returns: 
       Xunm: the unmixed stereo block resulting from the updated coefficients 
       coeffs: The updated coefficients 
       state0, state1 : the new filter states
    """
    # Simple online optimization, using random directions optimization:
    # Old values 0:
    Xunm0, state00, state10 = unmixing(
        coeffs, Xblock, state0, state1)  # starting point
    negabskl0 = abskl(Xunm0)
    # P=pdf(Xunm0[:,0],100); Q=pdf(Xunm0[:,1],100); negabskl0=-kldivergence(P,Q)-kldivergence(Q,P)
    #
    # global negabskl0
    # print("Block m=", m, "negabskl=", negabskl0)
    # Small random variation of coefficients, -0.05..0.05 for attenuations, -0.5..0.5 for delays:
    # coeffweights=np.array([0.1,0.1,1.0,1.0])*0.8
    global coeffweights
    for m in range(2):
        # coeffvariation=(np.random.rand(4)-0.5)*coeffweights
        coeffvariation = 4*(np.random.rand(4)-0.5) * \
            coeffweights  # small variation
        # coeffvariation=np.random.normal(loc=0.0, scale=0.5, size=4) #Gaussian distribution
        # new values 1:
        Xunm1, state01, state11 = unmixing(
            coeffs+coeffvariation, Xblock, state0, state1)
        negabskl1 = abskl(Xunm1)
        # P=pdf(Xunm1[:,0],100); Q=pdf(Xunm1[:,1],100); negabskl1=-kldivergence(P,Q)-kldivergence(Q,P)
        #
        if negabskl1 < negabskl0:  # New is better
            print("negabskl1, negabskl0=", negabskl1, negabskl0)
            negabskl0 = negabskl1
            coeffs = coeffs+coeffvariation
            Xunm = Xunm1
            state0 = state01
            state1 = state11
            # coeffweights=0.99*coeffweights+0.01*abs(coeffvariation)
            # limit the delays to maxdelay
            coeffs[2:] = np.clip(coeffs[2:], 0, maxdelay)
            print("coeffs=", coeffs)
            print("coeffweights=", coeffweights)
        else:  # old is better
            # negabskl=negabskl0
            Xunm = Xunm0
            state0 = state00
            state1 = state10
    # print("coeffs=", coeffs)
    # print("negabskl=", negabskl)
    # print("Coeffweights=", coeffweights)
    # End simple online optimization

    # scipy.optimize.minimize: too slow
    # coeffs_min = opt.minimize(minabsklcoeffs, coeffs, args=(X, state0, state1, maxdelay), method='CG',options={'disp':True, 'maxiter': 2})
    # coeffs=coeffs_min.x
    # print("coeffs=", coeffs)
    # Xunm, state0, state1 =unmixing(coeffs, Xblock, state0, state1, maxdelay)
    return Xunm, coeffs, state0, state1


coeffweights = np.array([0.1, 0.1, 1.0, 1.0])*0.4

# run it from file:
if __name__ == '__main__':
    import scipy.io.wavfile as wav
    # import scipy.optimize as opt
    import os
    import matplotlib.pyplot as plt

    # samplerate, X = wav.read("stereomoving.wav")
    # samplerate, X = wav.read("stereo_record_14.wav")
    #samplerate, X = wav.read("stereovoices.wav")
    samplerate, X = wav.read("mix16000.wav")
    # samplerate, X = wav.read("testfile_IDMT_dual_mic_array.wav")
    # samplerate, X = wav.read("2_channel_test_audio.wav")
    # samplerate, X = wav.read("/home/schuller/Nextcloud/Teaching/GeraldsBSS/7_channel_test_audio5.wav")
    # X=X[:,3::3]
    print("X.shape=", X.shape)
    # samplerate, X = wav.read("stereovoicemusic4.wav")
    # samplerate, X = wav.read("Schmoo.wav")
    # samplerate, X = wav.read("s_chapman.wav")
    # samplerate, X = wav.read("rockyou_stereo.wav")
    # samplerate, X = wav.read("sepchanstereo.wav")
    # samplerate, X = wav.read("stereotest2.wav")
    X = X*1.0/np.max(abs(X))

    os.system('espeak -s 120 "The original signal"')
    playsound(X*2**15, samplerate, 2)

    Blocksize = 1024
    M = 8  # number of blocks in signal memory
    Blocks = max(X.shape)//Blocksize  # total no. of blocks in signal
    print("Blocks=", Blocks)

    # Test length of delay filter coefficients:
    """
   a,b=allp_delayfilt(5.0)
   a=np.append(a,np.zeros(maxdelay+2-len(a)))
   b=np.append(b,np.zeros(maxdelay+2-len(b)))
   print("Delayfilter  a,b=", a , b)
   impulse_train = np.zeros(11)
   impulse_train[0] = 1
   h = scipy.signal.lfilter(b, a, impulse_train)   # impulse response
   plt.plot(h)
   plt.title('Allpass Delay Filter Impulse Response')
   plt.show()
   """

    # Initialize the filter memory:
    # state0=np.zeros(maxdelay+1)
    # fractional delay filter states for the length of the signal memory (M Blocks)
    state0 = np.zeros((maxdelay+1, M+1))
    # state1=np.zeros(maxdelay+1)
    state1 = np.zeros((maxdelay+1, M+1))

    # coeffs=[0.8, 0.8, 5.0, 5.0]
    # Staring value vector
    coeffs = [1.0, 1.0, 1.0, 1.0]
    negabskl0 = 0.0
    X_del = np.zeros(X.shape)
    sigmemory = np.zeros((Blocksize*M, 2))

    # Loop over the signal blocks:
    for m in range(Blocks):
        print("Block m=", m)
        Xblock = X[m*Blocksize+np.arange(Blocksize), :]
        # shift old samples left
        sigmemory[:-Blocksize, :] = sigmemory[Blocksize:, :]
        sigmemory[-Blocksize:, :] = Xblock  # Write new block on right end

        # Xunm, coeffs, state0, state1 = blockseparationoptimization(coeffs, Xblock, state0, state1)
        # shift states 1 left to make space for the newest state
        state0[:, :-1] = state0[:, 1:]
        state1[:, :-1] = state1[:, 1:]
        Xunm, coeffs, state0[:, -1], state1[:, -1] = blockseparationoptimization(
            coeffs, sigmemory, state0[:, -M-1], state1[:, -M-1])

        # Store newest unmixed block:
        X_del[m*Blocksize+np.arange(Blocksize), :] = Xunm[-Blocksize:, :]
        # print("abskl(X_del)=", abskl(X_del))

    wav.write("sepchan0.wav", samplerate, np.int16(X_del[:, 0]*2**15))
    wav.write("sepchan1.wav", samplerate, np.int16(X_del[:, 1]*2**15))
    wav.write("sepchanstereo.wav", samplerate, np.int16(X_del*2**15))
    print("Written to sepchan0.wav, sepchan1.wav and stereo in sepchanstereo.wav")
    print("coeffs=", coeffs)

    plt.plot(X_del[:, 0])
    plt.plot(X_del[:, 1])
    plt.title('The unmixed channels')
    plt.show()

    X_del = X_del*1.0/np.max(abs(X_del))
    os.system('espeak -s 120 "Separated Channel 0"')
    playsound(X_del[:, 0]*2**15, samplerate, 1)
    os.system('espeak -s 120 "Separated Channel 1"')
    playsound(X_del[:, 1]*2**15, samplerate, 1)
