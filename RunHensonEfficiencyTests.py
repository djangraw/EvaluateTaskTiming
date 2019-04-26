#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
RunHensonEfficiencyTests.py
* Recreate tests from Henson, 2007, figures 15.10,12,13, which compare the  
  relative efficiency of several designs.
* Note that in tests 2 & 3, the efficiency is multiplied by 4000 to bring it up 
  to the same order of magnitude as reported in the Henson chapter. The units
  and scaling are arbitrary as they depend on the scaling of the regressors.
 
References:
* Henson, R. (2007). Efficient experimental design for fMRI. Statistical 
  parametric mapping: The analysis of functional brain images, 193-210.

-Created 4/18/19 by DJ.
-Updated 4/19/19 by DJ - implemented tests for Henson Figures 15.12 and 15.13.
-Updated 4/26/19 by DJ - changed EvaluateTaskTiming.py name to effigenia
"""

# %% Import Packages
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from effigenia import EvaluateTaskTiming, LoadAfniTimingFile, ConvolveHRF, ApplyButterFilter, GetDesignEfficiency
afniDir='/Users/jangrawdc/abin' # modify to point to AFNI directory on your computer!
# Specify where images should be saved
outDir='HensonEfficiencyTest' 

effScale = 40 # E is unitless, so rescale values to match paper 

# %% TEST #1 (Figure 15.10)

# Set up task timing for various designs
nT=64;
nDesigns=6
pEvent=np.zeros((nT,nDesigns));
pEvent[np.arange(7,nT,8),0] = 1;
pEvent[:,1] = 0.5
pEvent[:,2] = np.cos(2.*np.pi/8.*np.arange(nT))/2+0.5
pEvent[:,3] = np.cos(2.*np.pi/16.*np.arange(nT))/2+0.5
pEvent[:,4] = np.cos(2.*np.pi/64.*np.arange(nT))/2+0.5
pEvent[:nT/2,5] = 1
# set up design descriptions
labels = ['fixed deterministic','stationary stochastic','dynamic stochastic (rapid)',
          'dynamic stochastic (intermediate)','dynamic stochastic (slow)','variable deterministic']

# Plot task designs
plt.figure(1510,figsize=(8,12), dpi= 80, facecolor='w', edgecolor='k');
plt.clf();
for i in range(nDesigns):
    plt.subplot(nDesigns,2,i*2+1)
    plt.bar(np.arange(nT),pEvent[:,i],1)
    plt.xlim([0,nT-1])
    plt.xlabel(labels[i])
    plt.ylim([0,1])
    if i==0:
        plt.title('occurrence probabilities')
plt.tight_layout()    


# === Run tests

# Set parameters to match paper text
nRand = 1;
TR = 1; 
runTime = nT*TR;
hpfCutoff = 120 # not specified in Henson figure, but implied elsewhere in text


# Make the directory if it doesn't exist
if not os.path.exists(outDir):
    os.mkdir(outDir);

# initialize results array
designEfficiency = np.zeros((nDesigns,nRand))

# execute script for each option
for i in range(nDesigns):
    print('***** version %d *****'%i)
    for j in range(nRand):
    
        # Create timing file
        with open("01_ev1.1D","w") as f:
            for t in range(nT):
                if np.random.rand()<pEvent[t,i]:
                    f.write('%d '%t)
                    
        # Evaluate timing
        print('*** version %d, iter %d ***'%(i,j))
        _,_,E,_ = EvaluateTaskTiming(evFiles=['01_ev1.1D'],fileType='afni_timing',afniDir=afniDir,
                            runTime=runTime,TR=TR,outFile='',outImagePrefix='',
                            hpfCutoff=hpfCutoff)

        # Add results 
        if np.array_equal(np.unique(pEvent[:,i]), np.array([0,1])): # if deterministic...
            designEfficiency[i,:] = E[0]*effScale; # fill in the whole row
            break                   # and stop, we only need to do this once
        else:
            designEfficiency[i,j] = E[0]*effScale;

# clean up
os.remove("01_ev1.1D")


# ===Plot results

# Plot mean efficiency of each design as a bar
plt.subplot(1,2,2); plt.cla();
plt.barh(np.arange(nDesigns)+1, np.mean(designEfficiency,axis=1),0.2) # to match Henson figure
plt.ylim([0.75,6.25])
plt.gca().invert_yaxis() # to align properly with designs on left-hand subplots
plt.tight_layout() 
# annotate plot
plt.title('Efficiency')
if nRand>1:
    plt.xlabel('mean across %d randomizations'%nRand) 
    # save figure
    plt.savefig('%s/DesignTypeEfficiency_%drand.png'%(outDir,nRand))
else:
    plt.savefig('%s/HensonFig15.10_EventProbTest.png'%(outDir))


# %% TEST #2 (Figure 15.12)

# Set parameters to match paper text
nT = 80;
TR = 1; 
runTime = nT*TR;
hpfCutoff = -1 # not specified in Henson figure. Results seem to match best when no filter is used.
filterOrder = 5
C = np.array([1,0]) # contrast matrix

# Set up figure
plt.figure(1512,figsize=(12,8), dpi= 80, facecolor='w', edgecolor='k');
plt.clf();

# Set up output matrices
nDesigns = 3
allCorr = np.zeros(nDesigns);
allEff = np.zeros(nDesigns);

# === Run tests
for i in range(nDesigns):
    # define responses
    if i==0:
        # resp always 4s after stim
        stimTimes = np.arange(0,80,8) # stimulus times
        respTimes = stimTimes+4
    elif i==1:
        # resp 0-8s after stim
        tNow = 0
        stimTimes = np.array([])
        respTimes = np.array([])
        while tNow<runTime:
            stimTimes = np.append(stimTimes,tNow)
            tNow = tNow+np.random.rand()*8
            respTimes = np.append(respTimes,tNow)
            tNow = tNow+np.random.rand()*8
    elif i==2:
        stimTimes = np.arange(0,80,8) # stimulus times
        # resp 4s after stim, 50% chance of resp on each trial
#        respTimes = np.array(0)
#        for t in stimTimes:
#            if  np.random.rand()<0.5:
#                respTimes = np.append(respTimes,t+4)
        respTimes = stimTimes[[0,1,3,7,8]]+4 # hard-code random choices to match figure
                
    # Create timing files
    # Make stim regressor
    with open("01_ev1.1D","w") as f:
        for t in stimTimes:
            f.write('%d '%t)
    with open("02_ev2.1D","w") as f:
        for t in respTimes:
            f.write('%d '%t)

    evFiles = ['01_ev1.1D','02_ev2.1D']
    evNames = ['Stimulus','Response']
    # Loop through each timing file and add its info to a DataFrame
    allTimings = pd.DataFrame({'trial_type':[],'onset':[],'duration':[]});
    for j,fname in enumerate(evFiles):        
        # Import timing file for one event type
        timings,nRuns = LoadAfniTimingFile(fname,runTime,afniDir);
        # add event names
        timings['trial_type'] = evNames[j]
        # add this event type's info to dataframe
        allTimings = pd.concat([allTimings,timings],sort=False)

    # Reset indices to be range(0,nEvents)
    allTimings = allTimings.reset_index(drop=True)
    
    # Convolve event times with HRF and sample at TRs
    Nvols = int(runTime*nRuns/TR);
    design,_ = ConvolveHRF(allTimings,Nvols,TR,eventTypes=evNames);
    M = design[:,:-1] # exclude constant term
    
    # Plot design        
    t = np.arange(Nvols)*TR;
    for k in range(len(evFiles)):
        plt.subplot(2,nDesigns*2,2*i+k+1)
        plt.plot(M[:,k],t); 
        plt.gca().invert_yaxis()
        # Annotate plot
        plt.title(evNames[k])
        if k==0:
            plt.ylabel('Time (s)')
            
    # apply HPF
    if hpfCutoff>0:
        Mfilt = ApplyButterFilter(M,1./hpfCutoff, fs=1./TR, btype='high',order=filterOrder)
    else:
        Mfilt = M;
        
    # Use contrast to calculate efficiency
    E = GetDesignEfficiency(Mfilt,C)
    E = GetDesignEfficiency(M,C)
    
    corrVal = np.corrcoef(Mfilt[:,0],Mfilt[:,1])

    # store values
    allEff[i] = E[0]*effScale # E is unitless, so rescale to match paper
    allCorr[i] = corrVal[0,1] # store correlation between the two (upper-right value)


    # Add values to figure
    plt.figtext(0.18+0.34*i,0.5,'corr = %.3g\nefficiency = %.3g'%(allCorr[i],allEff[i]),
                horizontalalignment='center') # E is unitless, so rescale to match paper
    
    # Print values
    print('corr = %.3g'%allCorr[i]);
    print('efficiency = %.3g'%allEff[i]) # E is unitless, so rescale to match paper
    
# clean up
os.remove("01_ev1.1D")
os.remove("02_ev2.1D")

    
# === Plot corr & efficiency of each design as a bar
# Plot correlation
plt.subplot(2,2,3); plt.cla();
plt.bar(np.arange(nDesigns)+1, allCorr) # to match Henson figure
# annotate plot
plt.title('Correlation between regressors')
plt.xlabel('Timing Design')
plt.ylabel('correlation')

# Plot efficiency
plt.subplot(2,2,4); plt.cla();
plt.bar(np.arange(nDesigns)+1, allEff) # to match Henson figure
# annotate plot
plt.title('Efficiency')
plt.xlabel('Timing Design')
plt.ylabel('efficiency')
plt.tight_layout()  
plt.subplots_adjust(hspace=0.3);
# save figure
plt.savefig('%s/HensonFig15.12_WorkingMemTest.png'%(outDir))



# %% TEST #3 (Figure 15.13)

# Set parameters to match paper text
nT = 160;
TR = 1; 
runTime = nT*TR;
hpfCutoff = -1 # not specified in Henson figure. Results seem to match best when no filter is used.
filterOrder = 5
C = np.array([1,0]) # contrast matrix

epochTimes = np.array([40,120])
epochDur = 40

# Set up figure
plt.figure(1513,figsize=(8,8), dpi= 80, facecolor='w', edgecolor='k');
plt.clf();

# Set up output matrices
nDesigns = 2;
allCorr = np.zeros(nDesigns);
allEff = np.zeros(nDesigns);

# === Run tests
for i in range(nDesigns):
    # define responses
    if i==0:
        # resp at 4s intervals during epoch
        stimTimes = np.append(np.arange(epochTimes[0],epochTimes[0]+epochDur,4), 
                              np.arange(epochTimes[1],epochTimes[1]+epochDur,4)); # stimulus times
    elif i==1:
        # 10 events per epoch, distributed randomly over 2s SOAs.
        stimTimes = np.array([]);
        for eTime in epochTimes:
           okStimTimes = np.arange(eTime,eTime+epochDur,2);
           epochStimTimes = np.random.choice(okStimTimes,10,False);
           stimTimes = np.append(stimTimes,epochStimTimes);

                
    # Create timing files
    # Make stim regressor
    with open("01_ev1.1D","w") as f:
        for t in stimTimes:
            f.write('%d '%t)
    with open("02_ev2.1D","w") as f:
        for t in epochTimes:
            f.write('%d:%d '%(t,epochDur))

    evFiles = ['01_ev1.1D','02_ev2.1D']
    evNames = ['Events','Epochs']
    # Loop through each timing file and add its info to a DataFrame
    allTimings = pd.DataFrame({'trial_type':[],'onset':[],'duration':[]});
    for j,fname in enumerate(evFiles):        
        # Import timing file for one event type
        timings,nRuns = LoadAfniTimingFile(fname,runTime,afniDir);
        # add event names
        timings['trial_type'] = evNames[j]
        # add this event type's info to dataframe
        allTimings = pd.concat([allTimings,timings],sort=False)

    # Reset indices to be range(0,nEvents)
    allTimings = allTimings.reset_index(drop=True)
    
    # Convolve event times with HRF and sample at TRs
    Nvols = int(runTime*nRuns/TR);
    design,_ = ConvolveHRF(allTimings,Nvols,TR,eventTypes=evNames);
    M = design[:,:-1] # exclude constant term
    
    # Plot design        
    t = np.arange(Nvols)*TR;
    for k in range(len(evFiles)):
        plt.subplot(2,nDesigns*2,2*i+k+1)
        plt.plot(M[:,k],t); 
        plt.gca().invert_yaxis()
        # Annotate plot
        plt.title(evNames[k])
        if k==0:
            plt.ylabel('Time (s)')
            
    # apply HPF
    if hpfCutoff>0:
        Mfilt = ApplyButterFilter(M,1./hpfCutoff, fs=1./TR, btype='high',order=filterOrder)
    else:
        Mfilt = M;
        
    # Use contrast to calculate efficiency
    E = GetDesignEfficiency(Mfilt,C)
    E = GetDesignEfficiency(M,C)
    
    corrVal = np.corrcoef(Mfilt[:,0],Mfilt[:,1])

    # store values
    allEff[i] = E[0]*effScale # E is unitless, so rescale to match paper's order of magnitude
    allCorr[i] = corrVal[0,1] # store correlation between the two (upper-right value)


    # Add values to figure
    plt.figtext(0.25+0.5*i,0.5,'corr = %.3g\nefficiency = %.3g'%(allCorr[i],allEff[i]),
                horizontalalignment='center') # E is unitless, so rescale to match paper
    
    # Print values
    print('corr = %.3g'%allCorr[i]);
    print('efficiency = %.3g'%allEff[i]) # E is unitless, so rescale to match paper
    
# clean up
os.remove("01_ev1.1D")
os.remove("02_ev2.1D")

    
# === Plot corr & efficiency of each design as a bar
# Plot correlation
plt.subplot(2,2,3); plt.cla();
plt.bar(np.arange(nDesigns)+1, allCorr) # to match Henson figure
# annotate plot
plt.title('Correlation between regressors')
plt.xlabel('Timing Design')
plt.ylabel('correlation')

# Plot efficiency
plt.subplot(2,2,4); plt.cla();
plt.bar(np.arange(nDesigns)+1, allEff) # to match Henson figure
# annotate plot
plt.title('Efficiency')
plt.xlabel('Timing Design')
plt.ylabel('efficiency')
plt.tight_layout()  
plt.subplots_adjust(hspace=0.3);
# save figure
plt.savefig('%s/HensonFig15.13_EpochTest.png'%(outDir))
