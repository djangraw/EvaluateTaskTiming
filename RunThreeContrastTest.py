#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
RunThreeContrastTest.py

Test EvaluateTaskTiming's % BOLD signal required metric using two boxcar 
regressors with varying rest periods, and three contrasts between them, as 
described in (Smith, 2007; Fig. 4).
* Note that the HRF and high-pass filter used in EvaluateTaskTiming do not 
  match Smith's exactly, so there will be some differences in the results.

-Created 4/5/19 by DJ.
"""

# %% Import Packages
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from EvaluateTaskTiming import EvaluateTaskTiming
afniDir='/Users/jangrawdc/abin' # modify to point to AFNI directory on your computer!


# %% 2 Block types of varying duration (Smith, 2007; Fig. 5)

# Declare constants to match (Smith, 2007)
tStim = 30; # duration of each block
tRests = np.arange(0,61,3); # duration of rest between each pair of blocks
# Set parameters to match paper text
tRun = 60*10; # duration of run (inferred from graph)
TR = 3.;
nT = tRun/TR
tCrit = 5.5;
hpfCutoff = 100;
#partMethod = 'beckmann'; # this method matches graph poorly
partMethod = 'none';      # this matches graph slightly better

# Specify where images should be saved
outDir='ThreeContrastTest' 

# Make the directory if it doesn't exist
if not os.path.exists(outDir):
    os.mkdir(outDir);

# Write contrast file corresponding to (Smith, 2007)
with open("00_conFile.txt","w") as f:
    f.write("1 0\n1 -1\n1 1")

# initialize results dataframe
allResults = pd.DataFrame(np.zeros((len(tRests),4)),columns=['tRest','pctBoldReq00','pctBoldReq01','pctBoldReq02']);

# Loop through options
for i,tRest in enumerate(tRests):
    # Create timing files
    with open("01_ev1.1D","w") as f:
        tNow = tRest;
        while tNow<tRun:
            f.write('%d:%d '%(tNow,tStim))
            tNow = tNow + 2*tStim + 2*tRest
            
    with open("02_ev2.1D","w") as f:
        tNow = 2*tRest + tStim;
        while tNow<tRun:
            f.write('%d:%d '%(tNow,tStim))
            tNow = tNow + 2*tStim + 2*tRest
    
    # Define which to plot
    if tRest in [0,15,60]:
        outImagePrefix='%s/tRest%d_'%(outDir,tRest)
    else:
        outImagePrefix=''

    # Evaluate timing
    print('***** tRest=%d *****'%tRest)
    _,_,_,B = EvaluateTaskTiming(evFiles=["01_ev1.1D","02_ev2.1D"],evNames=["ev1","ev2"],conFile="00_conFile.txt",
                                 fileType='afni_timing',afniDir=afniDir, runTime=tRun,TR=TR,
                                 tCrit=tCrit,outFile='',outImagePrefix=outImagePrefix,
                                 hpfCutoff=hpfCutoff,partMethod=partMethod)
    
    # Add results to dataframe
    allResults.loc[i,'tRest'] = tRest;
    allResults.loc[i,'pctBoldReq00'] = B[0];
    allResults.loc[i,'pctBoldReq01'] = B[1];
    allResults.loc[i,'pctBoldReq02'] = B[2];
        
        
# Plot results
plt.figure()
plt.plot(tRests,allResults.pctBoldReq00);
plt.plot(tRests,allResults.pctBoldReq01);
plt.plot(tRests,allResults.pctBoldReq02);
plt.legend(('C1: ev1','C2: ev1-ev2','C3: ev1+ev2'))
plt.ylabel('% BOLD Required')
# annotate plot
plt.xlabel('rest duration (s)')
plt.title('3-Contrast Tests')
# save figure
plt.savefig('%s/%s.png'%(outDir,outDir))

# clean up
os.remove("01_ev1.1D")
os.remove("02_ev2.1D")
os.remove("00_conFile.txt")
