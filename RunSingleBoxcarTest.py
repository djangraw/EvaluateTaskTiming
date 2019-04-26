#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
RunSingleBoxcarTest.py

Test effigenia's % BOLD signal required metric using a single boxcar 
regressor of varying duration, as described in (Smith, 2007; Fig. 3).
* Note that the HRF and high-pass filter used in EvaluateTaskTiming do not 
  match Smith's exactly, so there will be some differences in the results.

-Created 4/5/19 by DJ.
-Updated 4/26/19 by DJ - changed EvaluateTaskTiming.py name to effigenia
"""

# %% Import Packages
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
# import effigenia's functions even though file doesn't end in .py
import imp
eff = imp.load_source('effigenia', './effigenia')

# Declare constants
afniDir='/Users/jangrawdc/abin' # modify to point to AFNI directory on your computer!


# %% Run test and plot results

# Declare constants to match (Smith, 2007)
nTimepoints = np.array([50,100,200,400]) # duration of run in samples
tBoxcar = np.arange(3,200,3); # duration of each (on+off) block
# Set parameters to match paper text
TR = 3; 
tCrit = 5.5 
hpfCutoff = 50 
# Specify where images should be saved
outDir='OneBoxcarTest' 

# Make the directory if it doesn't exist
if not os.path.exists(outDir):
    os.mkdir(outDir);

# initialize results dataframe
allResults = pd.DataFrame(np.zeros((len(nTimepoints)*len(tBoxcar),3)),columns=['nT','tBoxcar','pctBoldReq']);

# execute script for each option
for i,nT in enumerate(nTimepoints):
    for j,tB in enumerate(tBoxcar):    
        # define run time
        tRun = nT*TR;
        
        # Create timing file
        with open("01_ev1.1D","w") as f:
            tNow = 0;
            while tNow<tRun:
                f.write('%d:%d '%(tNow,tB/2))
                tNow = tNow + tB;
                
        # Define which to plot
        if nT==200 and tB==40:
            outImagePrefix='%s/'%outDir
        else:
            outImagePrefix=''

        # Evaluate timing
        print('***** nT=%d, tBoxcar=%d *****'%(nT,tB))
        _,_,_,B = eff.EvaluateTaskTiming(evFiles=['01_ev1.1D'],fileType='afni_timing',afniDir=afniDir,
                            runTime=tRun,TR=TR,tCrit=tCrit,outFile='',outImagePrefix=outImagePrefix,
                            hpfCutoff=hpfCutoff)
        
        # Add results to dataframe
        iLine = i*len(tBoxcar) + j
        allResults.loc[iLine,'nT'] = nT;
        allResults.loc[iLine,'tBoxcar'] = tB;
        allResults.loc[iLine,'pctBoldReq'] = B;

# Plot results
plt.figure()
for j,nT in enumerate(nTimepoints):
    plt.plot(tBoxcar, allResults.loc[allResults.nT==nT,'pctBoldReq'],label='nT=%d'%nT);

# annotate plot
plt.xlabel('boxcar period (s)')
plt.ylabel('% BOLD Required')
plt.title('One-Boxcar Tests')
plt.legend()
# save figure
plt.savefig('%s/%s.png'%(outDir,outDir))
# clean up
os.remove("01_ev1.1D")
