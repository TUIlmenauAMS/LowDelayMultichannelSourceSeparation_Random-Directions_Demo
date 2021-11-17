#Program for optimizing (minimizing) using our algorithm of "random directions".
#With parallel processing for the random directions.
#usage: coeffmin=optimrandomdir(objfunction, coeffs, args=(X,))
#arguments: objfunction: name of objective function to minimize
#coeffs: Starting vector for the optimization, also determines the dimensionality of the input for objfunction
#Can be a tensor of any dimension!
#X: further arguments for objfunction
#returns: coeffmin: coeff vector which minimizes objfunction
#Includes line search for successful directions
#Gerald Schuller, May 2020

tracecoeffs=False #trace coefficients and writes them in coefftrace.pickle

def objfunccomp(objfunction, coeffs, args, bounds):
   #compact objective function for optimization
   if bounds !=():
         for n in range(len(bounds)):
            coeffs[n]=np.clip(coeffs[n],bounds[n][0], bounds[n][1])
   if len(args)==0:
      X0=objfunction(coeffs)
   elif len(args)==1:
      X0=objfunction(coeffs, args[0])
   else:
      X0=objfunction(coeffs, args)
   return X0

def optimrandomdir(objfunction, coeffs, args=(), bounds=(), coeffdeviation=1.0, iterations=1000, startingscale=2, endscale=0.0):
   #Reads in a block of a stereo signal, improves the unmixing coefficients,
   #applies them and returns the unmixed block
   #Arguments: objfunction: (char string) name of objective function to minimize
   #coeffs: array of the coefficients for which to optimize
   #args: Additional arguments for objfunction, must be a tuple! (in round parenthesis)
   #bounds: bounds for variables or coefficients, sequence (Tuple or list) of pairs of minima and maxima for each coefficient (optional)
   #This sequence can be shorter than the coefficent array, then it is only applied to the first coefficients that the bounds cover.
   #This can be used to include side conditions. Example: bounds=((0,1.5),(-1.57,1.57))
   #coeffdeviation: array of the expected standard deviation of the coefficients
   #iterations: number of iterations for optimization
   #startingscale: scale for the standard deviation for the random numbers at the beginning of the iterations 
   #endscale: scale of the random numbers et the end of the iterations.
   #returns: 
   #Xunm: the unmixed stereo block resulting from the updated coefficients 
   #coeffsmin: The coefficients which minimize objfunction
   
   import numpy as np
   import joblib #for parallel processing
   
   #print("args=", args, "args[0]=", args[0])
   sh=coeffs.shape #shape of objfunction input
   #Simple online optimization, using random directions optimization:
   #Initialization of the deviation vector:

   print("coeffdeviation=", coeffdeviation)
   try:
      if coeffdeviation==1.0:
         coeffdeviation=np.ones(sh)*1.0
         print("coeffdeviation=ones")
   except:
      print("Use specified coeffdeviation")	

   #Old values 0, starting point:
   
   if len(args)==0:
      X0=objfunction(coeffs)
   elif len(args)==1:
      X0=objfunction(coeffs, args[0])
   else:
      X0=objfunction(coeffs, args)
   """
   argstr=''
   for i in range(len(args)):
      argstr+=',args['+str(i)+']'
   argstr=argstr[1:]
   print('argstr=', argstr)
   print("eval(argstr)=", eval(argstr))
   #print("Call=", objfunction+'(coeffs'+argstr+')')
   #X0=eval(objfunction+'(coeffs'+argstr+')')
   #X0=objfunction(coeffs, eval(argstr))
   X0=objfunction(coeffs, args)
   """
   #Tracing coefficients:
   if tracecoeffs == True:
      import pickle
      coefftrace=np.expand_dims(coeffs, axis=2) #np.zeros(np.append(coeffs.shape,iterations))
   
   #Small random variation of coefficients, -0.05..0.05 for attenuations, -0.5..0.5 for delays:
   #coeffdeviation=np.array([0.1,0.1,1.0,1.0])*0.8
   #setfrac=8/max(sh)#0.0 <= setfrac <=1.0, 
               #probability of a coefficent to be updated, for reducing dimensionality. 
               #1.0: all coeffs are updated
               #A simple way to create subspaces of lower dimensionality
               #if setfrac is too small it easily gets stuck in local minima.
   #print("sh*2=", sh*2)
   #subspacematrix=np.random.normal(loc=0.0, scale=1.0, size=sh*2) #random matrix of size sh x sh, subspace approach
   m=0; mlastupdate=0
   parallelset=joblib.cpu_count() #number of available CPU's
   print("num of CPU's or parallelset:",parallelset) 
   #parallelset=2
   print("Number of parallel processes, parallelset= ", parallelset)
   print("endscale=", endscale)
   #iterations=int(100000/parallelset)  #for the optimization, 100000 for LDFB
   scale=8.0  #standard deviation for the gaussian random number generator
   scalehistsize=10
   scalehistory=np.zeros((scalehistsize,2))
   alpha=1/(2*len(coeffs)) #update factor for scale
   #alpha=1/(2*8)
   
   def functiontrial(coeffs, args, scale, size):
      coeffset=np.random.random(coeffs.shape)<=setfrac #Set of coefficients to be updated
      coeffvariation=np.random.normal(loc=0.0, scale=scale, size=sh)*coeffdeviation #Gaussian distribution
      coeffvariation *= coeffset #keep only the variation in the subset
      #coeffvariation=np.dot(subspacematrix,coeffvariation)  #subspace approach
      #print("coeffvariation=", coeffvariation)
      c1=coeffs+coeffvariation
      if bounds !=():
         for n in range(len(bounds)):
            c1[n]=np.clip(c1[n],bounds[n][0], bounds[n][1])
      #Here possibly loop over a set for parallel processing,
      #and X1 as the lowest over the set.
      #coeffmemory[parctr]=c1
      if len(args)==0:
         X1=objfunction(c1)
      elif len(args)==1:
         X1=objfunction(c1, args[0])
      else:
         X1=objfunction(c1, args)
      return X1, c1
   
   for m in range(0,iterations, parallelset): #count increases with number of parallel processes
   #while (np.linalg.norm(coeffdeviation)>1e-6):
      setfrac=6/max(sh) #take subspace of about 8 coefficients
      #setfrac=1.0 #full space
      #setfrac=np.random.rand(1) #random fraction of coefficients
      #scale=np.random.rand(1); #random scale (std. deviation for gauss distribution)
      #setfrac=np.clip(1.0-m/iterations,1/sh[0],1) #begin with global optimization, later fine tune coeff in lower dimensions
      #setfrac=setfrac **2 #become smaller faster
      #setfrac=setfrac ** 0.5 #become faster slower
      #scale=abs(np.random.normal(loc=0.0, scale=1.0, size=1)[0])
      #scale=np.random.rand(1)[0]
      #scale=((1.0-m/iterations))
      scale=np.clip((startingscale-endscale)*((1.0-m/iterations)**2)+endscale,1e-4,None)  #scale becomes smaller when iterations icrease (so far best)
      #scale=4.0*np.exp(-8*m/iterations)
      #scale*=0.9999  #exponential decay of scale of random numbers
      if m%(1000//parallelset)==0:
         print("m=", m, "setfrac=", setfrac, "scale=", scale ); m+=1
      #print("sh*2=", sh*2)
      #coeffvariation=(np.random.rand(4)-0.5)*coeffdeviation
      #coeffvariation=4*(np.random.random(sh)-0.5)*coeffdeviation  #small variation, uniform distribution
      #coeffset=np.ones(coeffs.shape)
      
      #parallel processing:
      X1memory=np.zeros(parallelset)# array of obtaine objective function results in the parallel list
      
      results = joblib.Parallel(n_jobs=parallelset)(joblib.delayed(functiontrial)(coeffs, args, scale=scale, size=sh) for parctr in range(parallelset))
      
      """
      for parctr in range(parallelset): #do parallel evaluations of new points
         
         X1memory[parctr]=X1
         #End parallel loop
      """
      #print("results=", results)
      i=0
      for res in results:
         X1memory[i]=res[0] #collecting the objective function values in a numpy array for finding the minimim
         i+=1
      #print("X1memory=", X1memory)
      minctr=np.argmin(X1memory) #index at which the objective function is minimum
      #print("minctr=", minctr)
      X1=X1memory[minctr]  #obtained minimum objective function value
      #print("X1=", X1)
      
      
      if X1<X0:  #New is better
         mlastupdate=m
         objimpr=(X0-X1)/X0
         X0=X1
         coeffvariation=results[minctr][1]-coeffs #successful coeff variation
         coeffset=(coeffvariation!=0.0) #set of changed coefficients
         coeffs=results[minctr][1] 
         #print("coeffs",coeffs)
         
         #line search:
         c1=coeffs
         #first towards larger:
         for lineexp in range(1,32): #-4,6
            X1=objfunccomp(objfunction, coeffs+2**lineexp * coeffvariation, args, bounds)
            if X1<X0:
               print("lineexp=",lineexp, "X1=", X1)
               X0=X1
               c1=coeffs+2**lineexp * coeffvariation
            else: #no improvement anymore
               break
         #now for smaller
         for lineexp in range(1,16): #-4,6
            X1=objfunccomp(objfunction, coeffs+0.5**lineexp * coeffvariation, args, bounds)
            if X1<X0:
               print("lineexp=",-lineexp, "X1=", X1)
               X0=X1
               c1=coeffs+0.5**lineexp * coeffvariation
            else:
               break #no improvent anymore
         
         coeffs=c1
         coeffdeviation+= coeffset*(-0.1*coeffdeviation+0.1*abs(coeffvariation)/np.sqrt(np.mean(coeffvariation**2)+1e-6))  #update deviation vector, normalize it
         coeffdeviation=np.clip(coeffdeviation,1e-1,None) #limit smallest value
         #coeffdeviation=np.clip(0.9*coeffdeviation+0.1*abs(coeffvariation),1e-6,None)  #update deviation vector
         #coeffdeviation += -0.1*coeffdeviation+0.1*abs(coeffvariation) #update deviation vector
         #coeffdeviation += coeffset*(-objimpr*coeffdeviation+objimpr*abs(coeffvariation)) #update deviation vector
         #coeffdeviation=coeffdeviation*0.1+0.1*np.abs(coeffs)
         #scale*=1.01
         #print("coeffs=", coeffs)
         
         #maxvec=np.argmax(np.dot(subspacematrix, coeffvariation))  #subspace approach
         #print("maxvec=", maxvec)
         #coeffvariation=coeffvariation/np.sqrt(np.dot(coeffvariation,coeffvariation)) #make norm(.)=1, subspace approach
         #subspacematrix[maxvec,:]=coeffvariation  #subspace approach
         print("coeffdeviation=", coeffdeviation)
         magvariation=np.sqrt(np.mean(coeffvariation**2))
         scalehistory[:-1,:]=scalehistory[1:,:] #shift up scalehistory
         #scalehistory[-1,0]=magvariation #std deviation for success
         scalehistory[-1,1]=objimpr  #obtained relative improvement of obj function
         scalehistarg=np.argmax(scalehistory[:,1]) #find index with largest improvement
         #scalehistarg=np.argmax(scalehistory[:,0]) #find index with largest std deviation
         #scale=3*scalehistory[scalehistarg,0]
         #scale=scale*(1-objimpr)+objimpr*magvariation
         #scale=scale*0.9+0.1*magvariation
         
         print("Obj. function X0=", X0, "iteration m=", m, "scale=", scale,"magvar.=", magvariation)#, "objimpr=", objimpr )
         #Trace coefficients:
         if tracecoeffs==True:
            coefftrace=np.concatenate((coefftrace, np.expand_dims(coeffs, axis=2)), axis=2)
      """
         #scale*=(1+8*alpha) #increase alpha, should balance out if every 8th iteration is a success
         #scale=2*magvariation
         scale*=(1+1.0*len(coeffs)*alpha)#increase alpha, should balance out if every len(coeffs) iteration is a success
      else:
         scale*=(1-alpha) #decrese scale slowly if no success
      """
      """
      elif(m-mlastupdate>100):
         scale*=0.99
         #scale=np.clip(scale*0.999,0.01,10)
         print("scale=", scale)
      """
   #End simple online optimization
   
   #scipy.optimize.minimize: too slow
   #coeffs_min = opt.minimize(minabsklcoeffs, coeffs, args=(X, state0, state1, maxdelay), method='CG',options={'disp':True, 'maxiter': 2})
   print("coeffdeviation=", coeffdeviation)
   print("coeffs=", coeffs)
   print("X0=", X0)
   if tracecoeffs==True:
      with open("coefftrace.pickle", 'wb') as coefftracefile:
         pickle.dump(coefftrace, coefftracefile)
      import matplotlib.pyplot as plt
      plt.plot(coefftrace[0,:,:].T)
      plt.show()
   
   #reading:
   #with open("coefftrace.pickle", 'rb') as coefftracefile:
       #print("Load robocoeffs.pickle")
       #coefftrace=pickle.load(coefftracefile)
       
   return coeffs
   

      
#coeffdeviation=1.0   #possible preset
#iterations=1000  #number of iterations for optimization, if not overridden locally
#startingscale=2.0  #scale for random numbers at the beginning of the iterations 
#endscale=0.0 #scale at the end of the iterations

#testing:
if __name__ == '__main__':
   import numpy as np
   from  scipy.special import jv #Bessel function

   xmin= optimrandomdir('jv', np.array([1.0]), (1.0,))
   #xmin= optimrandomdir('np.linalg.norm', np.array([1.0, 1.0]) )
   print("xmin=", xmin)

   
   
