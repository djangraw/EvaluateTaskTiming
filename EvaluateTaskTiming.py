#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
EvaluateTaskTiming.py

Reads timing files (.1D) from AFNI or design (.mat) or timing (3-column .txt) 
files from FSL and evaluates them using 4 different metrics: 
1) Mean square off-diagonal correlation between event regressors
2) Flatness of scree plot (normalized SVD eigenvalues of event regressor matrix)
3) Experimental Design Efficiency (based on Smith, 2007, eq. (2)
4) BOLD Signal Change Required for Significance at specified alpha value 
    (based on Smith, 2007, eq. (5)
Outputs the results as figures and a single-row text file for comparison across
task designs.

See python EvaluateTaskTiming.py -h for usage and input information.

References:
* Smith SM, Jenkinson M, Beckmann C, Miller K,
  Woolrich M. Meaningful design and contrast estimability
  in FMRI. Neuroimage 2007;34(1):127-36.

Created 2/21/19 by DJ.
Updated 2/25/19 by DJ - added efficiency and BOLD signal required calculations
Updated 3/8/19 by DJ - fixed eigenvalue calculation, cleaned up code
Updated 3/11/19 by DJ - renamed file, commented code
Updated 3/21/19 by DJ - added high-pass filtering of data
Updated 4/3/19 by DJ - added parent function callable from Python (not command line)
Updated 4/4/19 by DJ - added tight layout to plots to avoid overlapping labels
"""

# ==== Import packages ==== #

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import os.path
from scipy import signal
from scipy import stats
from scipy.linalg import null_space


# ==== Declare functions to load and preprocess event times ==== #

def LoadFslDesignFile(fname):
    """ Get the HRF-convolved event timecourse matrix from an FSL design matrix (.mat) file.
    
    Usage:
    design = LoadFslDesignFile(fname)
    
    Inputs:
    - fname   : filename of FSL formatted design (.mat) file

    Output:
    - design  : Matrix where each row is a TR and each column is an event type
    
    """
    
    # Read in columnar file (blank line -> row of nan's to indicate a new run has begun)
    dfDesign = pd.read_csv(fname,header=None,comment='/',delim_whitespace=True,skip_blank_lines=True)
    design = dfDesign.values
    
    return design
    

def LoadFslTimingFile(fname,runTime):
    """ Get the timings of various events in an FSL timing file.
    
    Usage:
    timings,nRuns = LoadFslTimingFile(fname,runTime)
    
    Inputs:
    - fname   : filename of FSL (3-column) formatted event timing file
    - runTime : time (sec) per run

    Output:
    - timings : pandas dataframe incluing timing of each event (in sec)
    - nRuns   : number of runs detected in timing file
    
    """
    
    # Read in columnar file (blank line -> row of nan's to indicate a new run has begun)
    timings = pd.read_csv(fname,header=None,delim_whitespace=True,skip_blank_lines=False,names=['onset','duration','amplitude'])
    # Set trial type
    eventName = os.path.splitext(os.path.basename(fname))[0];
    timings['trial_type'] = eventName
    
    # Adjust timings 
    iRun = 0;
    for i in range(timings.shape[0]):
        if pd.isna(timings.loc[i,'onset']): # row of nan's indicates a new run has begun
            iRun = iRun + 1;
        else: # adjust onset by adding duration of previous runs
            timings.loc[i,'onset'] = timings.loc[i,'onset'] + runTime*iRun;
        
    # Remove nan lines
    timings = timings[pd.notna(timings['onset'])]
    
    # Return results        
    nRuns = iRun+1
    return timings,nRuns


def LoadAfniTimingFile(fname,runTime,afniDir):
    """ Get the timings of various events in an AFNI timing file.
    
    Usage:
    timings,nRuns = LoadAfniTimingFile(fname,runTime,afniDir)
    
    Inputs:
    - fname   : filename of FSL (3-column) formatted event timing file
    - runTime : time (sec) per run
    - afniDir : path to afni installation directory (where timing_tool.py resides)

    Output:
    - timings : pandas dataframe incluing timing of each event (in sec)
    - nRuns   : number of runs detected in timing file
    
    """
    
    # Make sure AFNI is installed and in the given path
    if not os.path.isfile("%s/timing_tool.py"%afniDir):
        print("=== Using AFNI timing files requires an AFNI installation, specifically timing_tool.py.and its dependencies. ===")
        print("=== AFNI's timing_tool.py was not found in the specified directory: %s ==="%afniDir)
        raise ValueError("File %s/timing_tool.py not found!"%afniDir);
        
    # Create temporary columnar file to read in with Pandas
    tempFile = "TEMP.txt"
    # Convert timing file to columnar file
    os.popen("%s/timing_tool.py -multi_timing %s -multi_timing_to_event_list GE:tdm %s -verb 0"%(afniDir,fname,tempFile));
    # Read in columnar file (blank line -> row of nan's to indicate a new run has begun)
    timings = pd.read_csv(tempFile,header=None,delim_whitespace=True,skip_blank_lines=False,names=['onset','duration','amplitude'])
    os.remove(tempFile) # delete temporary file
    # Set trial type
    eventName = os.path.splitext(os.path.basename(fname))[0];
    timings['trial_type'] = eventName
    
    # Adjust timings 
    iRun = 0;
    for i in range(timings.shape[0]):
        if pd.isna(timings.loc[i,'onset']): # row of nan's indicates a new run has begun
            iRun = iRun + 1;
        else: # adjust onset by adding duration of previous runs
            timings.loc[i,'onset'] = timings.loc[i,'onset'] + runTime*iRun;
    
    # If no amps are given, default to 1
    timings.loc[pd.isna(timings['amplitude']),'amplitude']=1
    
    # Remove nan lines
    timings = timings[pd.notna(timings['onset'])]
     
    # Return results
    nRuns = iRun
    return timings,nRuns



# ==== Declare functions to create and preprocess event matrices ==== #

def ConvertToTimecourses(timings, Nvols, TR, microtime=0.1, eventTypes=[]):
    """ Create an event matrix from event timings; DO NOT convolve it with an HRF.
    
    Usage:
    design,eventTypes = ConvertToTimecourses(timings, Nvols, TR, microtime=0.1, eventTypes=[])
    
    Inputs:
    - timings : Design matrix, to be partitioned. Row = time, col = EV
    - Nvols   : Contrast that will define the partitioning. Row = EV, col = contrast
    - TR      : repetition time of MRI scan (sec)
    - microtime : resolution (sec) at which timecourse should be sampled before downsampling to TR
    - eventTypes:  List of event types (in order you specify) 
    
    Output:
    - design     : Matrix where each row is a TR and each column is an event type
    - eventTypes : List of event types (in alpha order if you didn't specify) 
    
    """
    # Set default duration to non-zero value
    timings.loc[timings['duration']==0,'duration'] = microtime;
    # Set default eventTypes to be sorted list of unique trial_type values
    if len(eventTypes)==0:
        eventTypes = pd.Series(timings['trial_type']).unique()
        eventTypes.sort() # put in alpha order
    # Set up
    design = []
    # Create EV and convolve with HRF
    for cond in eventTypes:
        fmri_stim = np.zeros(int(Nvols*TR/microtime), dtype=float)
        idx = timings['trial_type'] == cond
        start = (timings['onset'][idx]/microtime).astype(int)
        end   = (start + timings['duration'][idx]/microtime).astype(int)
        start = start.reset_index(drop=True)
        end   = end.reset_index(drop=True)
        for i, _ in enumerate(start):
            fmri_stim[start[i]:end[i]] = timings['amplitude'][i] # height
        design.append(fmri_stim[None].T)
    # Resample back to TR times (simple nearest neighbor interpolation)
    design = np.concatenate(design, axis=1)
    idx = np.round(np.arange(0, Nvols*TR, TR)/microtime).astype(int)
    design = np.concatenate([design[idx,:], np.ones((Nvols,1))], axis=1)
    # Return results
    return design,eventTypes


def ConvolveHRF(timings, Nvols, TR, microtime=0.1, eventTypes=[]):
    """ Create an event matrix from event timings and convolve it with an HRF.
    
    Usage:
    design,eventTypes = ConvolveHRF(timings, Nvols, TR, microtime=0.1, eventTypes=[])
    
    Inputs:
    - timings : Design matrix, to be partitioned. Row = time, col = EV
    - Nvols   : Contrast that will define the partitioning. Row = EV, col = contrast
    - TR      : repetition time of MRI scan (sec)
    - microtime : resolution (sec) at which timecourse should be sampled before downsampling to TR
    - eventTypes:  List of event types (in order you specify) 
    
    Output:
    - design     : Matrix where each row is a TR and each column is an event type
    - eventTypes : List of event types (in alpha order if you didn't specify) 
    
    """
    
    # Set default duration to non-zero value
    timings.loc[timings['duration']==0,'duration'] = microtime;
    # Set default eventTypes to be sorted list of unique trial_type values
    if len(eventTypes)==0:
        eventTypes = pd.Series(timings['trial_type']).unique()
        eventTypes.sort() # put in alpha order
    
    # HRF parameters (from FSL, reparameterized as function of k and theta)
    delay1 = 6.;  sigma1 = 2.449
    delay2 = 16.; sigma2 = 4.
    ratiogammas = 6;
    gammaK1 = delay1**2/sigma1**2; gammaT1 = sigma1**2/delay1;
    gammaK2 = delay2**2/sigma2**2; gammaT2 = sigma2**2/delay2;
 
    # Produce HRF 
    fmri_time = np.arange(0, Nvols*TR, microtime)
    fmri_gamm = stats.gamma.pdf(fmri_time, gammaK1, scale=gammaT1) - \
        stats.gamma.pdf(fmri_time, gammaK2, scale=gammaT2)/ratiogammas
    fmri_gamm = fmri_gamm/sum(fmri_gamm)
    fmri_pgamm = np.concatenate((np.zeros(np.shape(fmri_gamm)), fmri_gamm))
    fmri_fgamm = np.fft.fft(fmri_pgamm)
   
    # Set up
    design = []
    # Create EV and convolve with HRF
    for cond in eventTypes:
        fmri_stim = np.zeros(int(Nvols*TR/microtime), dtype=float)
        idx = timings['trial_type'] == cond
        start = (timings['onset'][idx]/microtime).astype(int)
        end   = (start + timings['duration'][idx]/microtime).astype(int)
        start = start.reset_index(drop=True)
        end   = end.reset_index(drop=True)
        for i, _ in enumerate(start):
            fmri_stim[start[i]:end[i]] = timings['amplitude'][i] # height
        fmri_pstim = np.concatenate((np.zeros(np.shape(fmri_stim)),fmri_stim))
        fmri_fstim = np.fft.fft(fmri_pstim)
        fmri_conv = np.real(np.fft.ifft(fmri_fstim * fmri_fgamm))
        fmri_conv = fmri_conv[0:np.shape(fmri_gamm)[0]]
        design.append(fmri_conv[None].T)
    # Resample back to TR times (simple nearest neighbor interpolation)
    design = np.concatenate(design, axis=1)
    idx = np.round(np.arange(0, Nvols*TR, TR)/microtime).astype(int)
    design = np.concatenate([design[idx,:], np.ones((Nvols,1))], axis=1)
    # Return results
    return design,eventTypes


# for any filtering of data
def GetButterFilter(cutoffs,fs,btype,order=5):
    """ Create a Butterworth filter.
    
    Usage:
    b,a = GetButterFilter(cutoffs,fs,btype,order=5)
    
    Inputs:
    - cutoffs : a scalar (if btype='high' or 'low') or 2-element list (if btype='band') containing the cutoff frequencies in Hz.
    - fs      : the sampling frequency in Hz
    - btype   : is the type of filter ('high' for high-pass, 'low' for low-pass, or 'band' for band-pass).
    - order   : the order of the filter.
    
    Outpuuts:
    - b : the numerator coefficients of the filter transfer function.
    - a : the denominator coefficients of the filter transfer function.
    """
    
    nyq = 0.5 * fs # nyquist frequency
    normal_cutoffs = cutoffs / nyq # normalize using nyquist freq
    b, a = signal.butter(order, normal_cutoffs, btype=btype, analog=False) # get filter coefficients
    return b,a


def ApplyButterFilter(data,cutoffs,fs,btype,order=5):
    """ Apply a Butterworth filter to data (using scipy.signal.filtfilt).
    
    Usage:
    y = ApplyButterFilter(data,cutoffs,fs,btype,order=5)
    
    Inputs:
    - data    : an array containing the data you want to filter (rows=time)
    - cutoffs : a scalar (if btype='high' or 'low') or 2-element list (if btype='band') containing the cutoff frequencies in Hz.
    - fs      : the sampling frequency in Hz
    - btype   : is the type of filter ('high' for high-pass, 'low' for low-pass, or 'band' for band-pass).
    - order   : the order of the filter.
    
    Outpuuts:
    - y : the filtered data.
    """
    
    b, a = GetButterFilter(cutoffs, fs, btype, order=order) # get filter coefficients
    y = signal.filtfilt(b, a, data, axis=0) # filter in both direction so as to introduce zero delays
    return y



def PalmPartition(M,C,meth):
    """
    Partition a design matrix into regressors of interest and
    nuisance according to a given contrast.
    
    Usage:
    X,Z,eCm,eCx = PalmPartition(M,C,meth,Y)
    
    Inputs:
    M    : Design matrix, to be partitioned.
    C    : Contrast that will define the partitioning.
    meth : Method for the partitioning. It can be:
           - 'Guttman'
           - 'Beckmann'
           - 'Ridgway'
           - 'none' (does nothing, X=M, Z=[])
    
    Outputs:
    X    : Matrix with regressors of interest.
    Z    : Matrix with regressors of no interest.
    eCm  : Effective contrast, equivalent to the original,
           for the partitioned model [X Z], and considering
           all regressors.
    eCx  : Same as above, but considering only X.
    
    References:
    * Guttman I. Linear Models: An Introduction. Wiley,
      New York, 1982.
    * Smith SM, Jenkinson M, Beckmann C, Miller K,
      Woolrich M. Meaningful design and contrast estimability
      in FMRI. Neuroimage 2007;34(1):127-36.
    * Ridgway GR. Statistical analysis for longitudinal MR
      imaging of dementia. PhD thesis. 2009.
    * Winkler AM, Ridgway GR, Webster MG, Smith SM, Nichols TE.
      Permutation inference for the general linear model.
      Neuroimage. 2014 May 15;92:381-97.
    _____________________________________
    A. Winkler, G. Ridgway & T. Nichols
    FMRIB / University of Oxford
    Mar/2012 (1st version)
    Aug/2013 (major revision)
    Dec/2015 (this version)
    Feb/2019 (ported from MATLAB to Python)
    http://brainder.org

    """
       
    # make comparisons case-insensitive
    meth = meth.lower()
    # enforce C=2d, M=2d
    if len(C.shape)==1:
        C = C.reshape(C.shape+(1,))
    if len(M.shape)==1:
        M = M.reshape(M.shape+(1,))
    
    # Perform main calculations
    if meth == 'guttman': # Partition as in (Guttman, 1982)
        idx   = np.any(C!=0,1);
        X     = M[:,idx];
        Z     = M[:,~idx];
        eCm   = np.vstack((C[idx,:],C[~idx,:]));
            
    elif meth == 'beckmann': # Partition as in (Smith, 2007)
        Cu    = null_space(C.T);
        D     = np.linalg.pinv(np.matmul(M.T, M));
        CDCi  = np.linalg.pinv(np.matmul(np.matmul(C.T, D), C));
        Pc    = np.matmul(np.matmul(np.matmul(C,CDCi),C.T),D);
        Cv    = Cu - np.matmul(Pc,Cu);
        F3    = np.linalg.pinv(np.matmul(np.matmul(Cv.T,D),Cv));
            
        X     = np.matmul(np.matmul(np.matmul(M, D), C), CDCi);
        Z     = np.matmul(np.matmul(np.matmul(M, D), Cv), F3);
        
        eCm = np.vstack((np.eye(X.shape[1]), 
                        np.zeros((Z.shape[1],X.shape[1]))));
    
        
    elif meth == 'ridgway': # Partition as in (Ridgway, 2009)
        rC    = np.linalg.matrix_rank(C);
        rM    = np.linalg.matrix_rank(M);
        rZ    = rM - rC;
        pinvC = np.linalg.pinv(C.T);
        C0    = np.eye(M.shape[1]) - np.matmul(C, np.linalg.pinv(C));

        tmpX  = np.matmul(M,pinvC);
        tmpZ,_,_  = np.linalg.svd(np.matmul(M,C0));
        # Calculate Z and X
        Z     = tmpZ[:,:rZ];
        X     = tmpX - np.matmul(np.matmul(Z, np.linalg.pinv(Z)),tmpX);
        # Calculate eCm
        eCm = np.vstack((np.eye(X.shape[1]),
                        np.zeros((Z.shape[1],X.shape[1]))));
        
    elif meth == 'none': # Don't partition the data. Return the input matrices.
        X     = M;
        Z     = [];
        eCm   = C;
        
    else:
        raise ValueError('"%s" - Unknown partitioning scheme'%meth);
    
    # Select X-relevant elements of eCm for eCx
    eCx = eCm[:X.shape[1],:];
            
    # Return result
    return X,Z,eCm,eCx


# ==== Declare functions to calculate design quality metrics ==== #

def GetDesignEfficiency(M,C):
    """ Get the efficiency of an experimental design given a contrast. Our
    Efficiency metric is a simplified version of Appendix B from Smith, 2007:
        E = (C'(M'M)C)^-0.5
    
    Usage:
    e = GetDesignEfficiency(M,C)
    
    Inputs:
    - M    : Design matrix, to be partitioned. Row = time, col = EV
    - C    : Contrast that will define the partitioning. Row = EV, col = contrast

    Output:
    - e    : Efficiency of design in estimating each contrast
    
    References: 
     * Smith SM, Jenkinson M, Beckmann C, Miller K,
       Woolrich M. Meaningful design and contrast estimability
       in FMRI. Neuroimage 2007;34(1):127-36.
    
    """
    
    # Enforce 2D arrays for both M and C
    if len(M.shape)==1:
        M = M.reshape(M.shape+(1,))
    elif len(M.shape)==3:
        M = M.reshape(M.shape[0:2])
        
    if len(C.shape)==1:
        C = C.reshape(C.shape+(1,))
    elif len(C.shape)==3:
        C = C.reshape(C.shape[0:2])
        
    # Calculate efficiency
    # E = (C'(M'M)C)^-0.5
    e = np.power( np.diagonal( np.matmul(np.matmul(C.T, np.matmul(M.T, M)), C)), -0.5)
    
    # Return result
    return e



def GetBoldSensitivityMetric(M, C, alpha=0.05, N=0.7):
    """ Get the BOLD sensitivity required to see an effect for a given design & 
    contrast. Based on equation 5 from Smith, 2007.
    
    Usage:
    B = GetBoldSensitivityMetric(M, C, alpha=0.05, N=0.7)
    
    Inputs:
    - M    : Design matrix, to be partitioned. Row = time, col = EV
    - C    : Contrast that will define the partitioning. Row = EV, col = contrast
    - alpha: alpha value used to select a t statistic cutoff for significance
    - N    : noise standard deviation expressed as a percentage fraction of the baseline intensity 

    Output:
    - B    : % BOLD signal required for a contrast to reach significance.
    
    References: 
     * Smith SM, Jenkinson M, Beckmann C, Miller K,
       Woolrich M. Meaningful design and contrast estimability
       in FMRI. Neuroimage 2007;34(1):127-36.
       
    """  
       
    # Enforce 2D arrays for both M and C
    if len(M.shape)==1:
        M = M.reshape(M.shape+(1,))
    elif len(M.shape)==3:
        M = M.reshape(M.shape[0:2])
        
    if len(C.shape)==1:
        C = C.reshape(C.shape+(1,))
    elif len(C.shape)==3:
        C = C.reshape(C.shape[0:2])
        
    # Get critical T value
    MC = np.matmul(M,C) # contrast timecourses
    dof = MC.shape[0]-np.linalg.matrix_rank(MC)
    t_c = stats.t.ppf(1-alpha, dof) # critical t value
    # get D (for each contrast): metric of design efficiency
    h = np.max(MC,0)-np.min(MC,0)
    e = np.power( np.diagonal( np.matmul(np.matmul(C.T, np.matmul(M.T, M)), C)), -0.5) # efficiency-related value
    D = h * e
    # Put in to formula
    B = t_c*D*N
    
    # Return result
    return B
    


# ==== Declare function for complete evaluation of task timing ==== #
    
def EvaluateTaskTiming(evFiles,evNames=[],conFile='',fileType='',afniDir='./',
                            runTime=0.,TR=2.,alpha=0.025,tCrit=-1,noiseStd=0.7,
                            outFile='out.txt',outImagePrefix='',pauseToReview=False,
                            skipHrf=False, partMethod='beckmann',
                            filterType='butterworth',filterOrder=5,hpfCutoff=50,lpfCutoff=-1):
    """ Calculate several figures of merit for a given set of experimental times and contrasts.
    
    Usage:
    vAvg,flatness,E,B = EvaluateTaskTiming(evFiles,evNames=[],conFile='',fileType='',afniDir='./',
                            runTime=0.,TR=2.,alpha=0.025,tCrit=-1,noiseStd=0.7,
                            outFile='out.txt',outImagePrefix='',pauseToReview=False,
                            skipHrf=False, partMethod='beckmann',
                            filterType='butterworth',filterOrder=5,hpfCutoff=50,lpfCutoff=-1)
    
    Inputs:
    - evFiles : list of event variable files
    - evNames : list of event variable names (if empty, use evFiles filenames)
    - conFile : name of the contrast text file (row = contrast, column = EV. If empty, each EV is used as a contrast.)
    - fileType: what kind of files are evFiles? (afni_timing, fsl_timing, or fsl_design currently supported.)
    - afniDir : directory where AFNI can be found (only required if AFNI timing files are used)
    - runTime : time per run
    - TR      :  MRI sequence repetition time (for event sampling)
    - alpha   : desired significance level (p value)
    - tCrit   : desired critical t value (overrides alpha)
    - noiseStd: estimated noise standard deviation (~0.7 for 3T)
    - outFile : text output filename
    - outImagePrefix : directory/start of filename where images should be saved  
    - skipHrf     : should we skip convolving EVs with a hemodynamic response function?
    - partMethod  : which method should be used to partition the data?
    - filterType  : which type of filter should be used on the EVs?
    - filterOrder : filter order
    - hpfCutoff   : high-pass filter time in seconds
    - lpfCutoff   : low-pass filter time in seconds

    Outputs:
    - vAvg     : mean squared correlation between EVs
    - flatness : flatness of SVD eigenvalue scree plot (mean(eigvals)/eigvals[0])
    - E        : efficiency of task design (one for each contrast)
    - B        : % BOLD signal required to get a significant result (given tCrit or alpha)
    - a text file containing these values is saved to outFile
    - images illustrating these evaluations are saved to outImagePrefix: 
        00-EvTimecourse.png, 01-EvCorr.png, 02-EigScree.png, 03-Efficiency.png, and 04-BoldRequired.png
    
    """ 
    
    # Declare defaults
    if len(evNames)==0: # set event names to event timing filenames
        evNames = [os.path.splitext(os.path.basename(fname))[0] for fname in evFiles];
        
    if len(conFile)==0: # set to be one contrast for each event type
        cInput = np.eye(len(evNames)); # assume each EV is a contrast
    else:
        cInput = pd.read_csv(conFile,delim_whitespace=True,header=None).values         
    # take transpose of contrast matrix input so row = EV, col = contrast
    C = cInput.T 
    # should results figures be plotted?
    doPlots = (len(outImagePrefix)>0); # only if an outImagePrefix is provided.
    
    
    # Display contrasts (as weighted combo of events) for user to check
    print('=== %d contrasts detected:'%C.shape[1])
    for i in range(C.shape[1]):
        printStr = 'contrast %d ='%(i+1)
        for j in range(C.shape[0]):
            if C[j,i] > 0:
                printStr = printStr + " + %d * %s"%(C[j,i],evNames[j])
            elif C[j,i] < 0:
                printStr = printStr + " - %d * %s"%(-C[j,i],evNames[j])
        print(printStr)

    # turn on interactive mode so showing a figure doesn't stop execution.
    plt.ion()
    
    
    # %% ===== Get HRF-convolved event timecourses
    
    # Import FSL design file
    if fileType == "fsl_design": # FSL design file was specified   
        M = LoadFslDesignFile(evFiles[0])
        Nvols = M.shape[0]
        
    # Otherwise, import a list of timing files
    else:
        # Loop through each timing file and add its info to a DataFrame
        allTimings = pd.DataFrame({'trial_type':[],'onset':[],'duration':[]});
        for i,fname in enumerate(evFiles):
            
            # Import timing file for one event type
            if fileType == "afni_timing": # AFNI timing files were specified
                timings,nRuns = LoadAfniTimingFile(fname,runTime,afniDir);
            elif fileType == "fsl_timing": # FSL timing files were specified             
                timings,nRuns = LoadFslTimingFile(fname,runTime);
            else:
                raise ValueError("Input fileType must be either fsl_design, afni_timing, or fsl_timing.")            
            # add event names
            timings['trial_type'] = evNames[i]
            # add this event type's info to dataframe
            allTimings = pd.concat([allTimings,timings],sort=False)
    
        # Reset indices to be range(0,nEvents)
        allTimings = allTimings.reset_index(drop=True)
        
        # Convolve event times with HRF and sample at TRs
        Nvols = int(runTime*nRuns/TR);
        if skipHrf:
            design,_ = ConvertToTimecourses(allTimings,Nvols,TR,microtime=TR,eventTypes=evNames);
        else:
            design,_ = ConvolveHRF(allTimings,Nvols,TR,microtime=0.1,eventTypes=evNames);
        M = design[:,:-1] # exclude constant term
      
    # %% ===== High-pass filter and partition EVs
    if filterType=='butterworth' or filterType=='butter':
        if lpfCutoff<0 and hpfCutoff<0:
            print ('Skipping filtering step...')
            Mfilt = M;
        elif lpfCutoff<0:
            print('High-pass filtering data at %.1f sec using Butterworth filter of order %d.'%(hpfCutoff,filterOrder))        
            Mfilt = ApplyButterFilter(M,1./hpfCutoff, fs=1./TR, btype='high',order=filterOrder)
        elif hpfCutoff<0:
            print('Low-pass filtering data at %.1f sec using Butterworth filter of order %d.'%(lpfCutoff,filterOrder))        
            Mfilt = ApplyButterFilter(M,1./lpfCutoff, fs=1./TR, btype='low',order=filterOrder)
        else:
            print('Band-pass filtering data at [%.1f, %.1f] sec using Butterworth filter of order %d.'%(lpfCutoff,hpfCutoff,filterOrder))        
            Mfilt = ApplyButterFilter(M,np.array([1./lpfCutoff, 1./hpfCutoff]), fs=1./TR, btype='band',order=filterOrder)
#        Mfilt = butter_highpass_filter(M, 1./hpfCutoff, fs=1./TR, order=filterOrder)
    elif filterType=='none':
        print ('Skipping filtering step...')
        Mfilt = M;
    else:
        raise ValueError('Filter type %s not recognized!'%filterType)
                
   # Partition matrix    
    X,_,_,eCx = PalmPartition(Mfilt,C,partMethod)

    # %% ===== Plot event regressors
    
    if doPlots:
        # Plot design
        plt.figure(figsize=(8,8), dpi= 80, facecolor='w', edgecolor='k')
        t = np.arange(Nvols)*TR;
        plt.subplot(3,1,1)
        plt.plot(t,M); 
        # Annotate plot
        plt.title('Event Variables (EVs)')
        plt.xlabel('time (s)')
        plt.ylabel('EV value')
        # Add legend
        plt.legend(evNames)
        
        # Plot filtered design
        plt.subplot(3,1,2)
        plt.plot(t,Mfilt); 
        # Annotate plot
        plt.title('Filtered EVs')
        plt.xlabel('time (s)')
        plt.ylabel('EV value')        
        # Add legend
        plt.legend(evNames)

        # Plot partitioned contrasts
        plt.subplot(3,1,3)
        plt.plot(t,np.matmul(X,eCx)); 
        # Annotate plot
        plt.title('Filtered Contrasts')
        plt.xlabel('time (s)')
        plt.ylabel('Contrast value')        
        # Add legend
        plt.legend(['C%d'%x for x in range(C.shape[1])])
        
        # Prevent subplot labels from overlapping
        plt.tight_layout()
        # Save & show results
        plt.savefig("%s00-EvTimecourse.png"%outImagePrefix)
        plt.show()
    
    
    # %% ===== Get metric #1: correlation
    
    # Make design matrix into pandas dataframe
    dfDesign = pd.DataFrame(Mfilt,columns=evNames); # remove constant column
    nEVs = len(evNames)
    # Get squared correlation matrix & its off-diagonal values
    corrMat = np.square(dfDesign.corr())
    corrVals = np.array(corrMat)[np.triu_indices(nEVs, k = 1)];
    if corrVals.size>0:
        vMax = np.max(corrVals)  # max off-diagonal value
        vAvg = np.mean(corrVals) # mean off-diagonal value
    else:
        vMax=1;
        vAvg=0;
    
    # Plot squared correlation matrix
    if doPlots:
        # Plot squared correlation matrix
        plt.figure(figsize=(4,4), dpi= 80, facecolor='w', edgecolor='k')
        plt.matshow(corrMat,vmin=0.,vmax=vMax*1.1)
        # Annotate plot
        plt.title('EV Squared Correlation matrix',y=1.1)
        plt.xticks(np.arange(len(evNames)),evNames)
        plt.yticks(np.arange(len(evNames)),evNames)
        plt.colorbar()
        # Prevent subplot labels from overlapping
        plt.tight_layout()
        # Save & show results
        plt.savefig("%s01-EvCorr.png"%outImagePrefix)
        plt.show()
    
    # Print summary metric
    print("=== Metric 1: mean square off-diagonal correlation = %.3g"%vAvg)
    
    
    # %% ===== Get metric #2: PCAs w/ flatness of scree plot
    
    # Get SVD eigenvalues (NOTE: These are not PCA eigenvalues.)
    U, S, V = np.linalg.svd(Mfilt) 
    eigvals = S / np.sum(S)  # normalize eigenvalues to sum to one
    
    # Plot eigenvalues as line graph
    if doPlots:
        # Make scree plot
        plt.figure(figsize=(6,4), dpi= 80, facecolor='w', edgecolor='k')
        xEig = np.arange(nEVs) + 1
        plt.plot(xEig, eigvals, 'ro-', linewidth=2)
        # Annotate plot
        plt.ylim(0,eigvals[0]*1.1)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        # Make legend
        plt.legend(['Eigenvalues from SVD'])
        # Prevent subplot labels from overlapping
        plt.tight_layout()
        # Save & show results
        plt.savefig("%s02-EigScree.png"%outImagePrefix)
        plt.show()
    
    # Calculate summary metric of flatness
    #   variance perfectly spread across EVs: flatness --> 1
    #   all variance in one EV: flatness --> 0
    flatness = np.mean(eigvals)/eigvals[0] 
    # Print summary metric
    print("=== Metric 2: flatness of scree plot = %.3g"%flatness)
    
    
    # %% ===== Get metric #3: efficiency
    
    # Use contrast to calculate efficiency
    E = GetDesignEfficiency(Mfilt,C)
    
    # Plot efficiency as bars
    if doPlots:
        # Plot results
        plt.figure(figsize=(6,4), dpi= 80, facecolor='w', edgecolor='k')
        xBar = np.arange(len(E))+1
        plt.bar(xBar,E)
        # Annotate plot
        plt.xticks(xBar)
        plt.title('Experimental Design Efficiency')
        plt.xlabel('Contrast')
        plt.ylabel('Efficiency')
        # Prevent subplot labels from overlapping
        plt.tight_layout()
        # Save & show results
        plt.savefig("%s03-Efficiency.png"%outImagePrefix)
        plt.show()
    
    # Print results
    print("=== Metric 3: Experimental Design Efficiency:")
    for i,e in enumerate(E):
        print("   Contrast %d: %.3g"%(i,e));              
    
    
    # %% ===== Get metric #4: % BOLD effect required to detect an effect
        
    # Get BOLD sensitivity metric
    if tCrit>0: # convert critical t value to alpha
        MC = np.matmul(X,eCx) # contrast timecourses
        dof = MC.shape[0]-np.linalg.matrix_rank(MC)
        alpha = 1-stats.t.cdf(tCrit,dof);
        
    B = GetBoldSensitivityMetric(X, eCx, alpha=alpha, N=noiseStd);
        
    # Plot BOLD sensitivity as bars
    if doPlots:
        # Plot results
        plt.figure(figsize=(6,4), dpi= 80, facecolor='w', edgecolor='k')
        xBar = np.arange(len(B))+1
        plt.bar(xBar,B)
        # Annotate plot
        plt.xticks(xBar)
        plt.title('BOLD Signal Change Required for Significance at alpha=%.2g'%alpha)
        plt.xlabel('Contrast')
        plt.ylabel('BOLD (%)')
        # Prevent subplot labels from overlapping
        plt.tight_layout()
        # Save & show results
        plt.savefig("%s04-BoldRequired.png"%outImagePrefix)
        plt.show()
    
    # Print results
    print('=== Metric 4: BOLD Signal Change Required for Significance at alpha=%.2g:'%alpha)
    for i,b in enumerate(B):
        print("   Contrast %d: %.3g%%"%(i,b))


    # %% ===== Write results to file
    
    if len(outFile)>0: # if we were given a file to write to
        print("=== Writing metrics to %s..."%outFile)
        with open(outFile,'w') as f:
            # Write headers
            f.write("MeanEvCorrSq ")
            f.write("eigFlatness ")
            f.write("".join(["efficiency%02d "%i for i in range(len(E))]))
            f.write(" ".join(["pctBoldReq%02d"%i for i in range(len(B))]))
            f.write("\n")
            # Write values
            f.write("%.3g "%vAvg)
            f.write("%.3g "%flatness)
            f.write("".join(["%.3g "%e for e in E]))
            f.write(" ".join(["%.3g"%b for b in B]))
            f.write("\n")
       
    # Display figures for user to review
    if doPlots:        
        plt.pause(0.01) # cue plots to render

    if pauseToReview:
        raw_input('=== Calculation complete! Press enter to close figures and exit:') # pause for user input

    # Finish
    print("=== Done! ===")    
    return vAvg,flatness,E,B



# %% ==== Set Up Input Argument Parser ==== #

parser = argparse.ArgumentParser(description='Evaluate experiment timing, saving figures ' + 
                                 'and text file with: \n' + 
                                 '(1) mean square EV intercorrelation, \n' +
                                 '(2) flatness of SVD scree plot of design matrix, \n' + 
                                 '(3) experimental design efficiency for each contrast, and \n' +
                                 '(4) BOLD signal change required for each contrast. \n'+
                                 'This output text file will contain each of these metrics as a column.')
parser.add_argument('--evNames', nargs='*', default='', help='event variable names')
parser.add_argument('--evFiles', nargs='*', default='', help='event variable files (or design file)')
parser.add_argument('--conFile', default='', help='file containing space-separated contrasts (row=contrast, col=event type)')
parser.add_argument('--fsl_design', action='store_true',help='specify evFiles is an FSL-style design file (usually called design.mat)')
parser.add_argument('--fsl_timing', action='store_true', help='specify evFiles are FSL-style 3-column timing files (empty line = end of run)')
parser.add_argument('--afni_timing', action='store_true', help='specify evFiles are AFNI-style timing files (one row per run)')
parser.add_argument('--afniDir', default='./',help='directory where afni is located')
parser.add_argument('--runTime', default=0, help='duration of each run (sec)')
parser.add_argument('--TR', default='2', help='MRI repetition time (sec)')
parser.add_argument('--alpha', default='0.025', help='desired significance level for BOLD signal change metric')
parser.add_argument('--tCrit', default='-1', help='desired critical t vaule for BOLD signal change metric (overrides --alpha)')
parser.add_argument('--noiseStd', default='0.7', help='estimated standard deviation (%%) of scanner noise (~0.7 at 3T)')
parser.add_argument('--outFile', default='out.txt', help='text file where summary values should go')
parser.add_argument('--outImagePrefix', default='', help='directory/start of filename where output images should go')
parser.add_argument('--pauseToReview', action='store_true', help='pause execution at the end to view figures')
parser.add_argument('--skipHrf', action='store_true', help='do not convolve event blocks with HRF')
parser.add_argument('--partMethod', default='beckmann', help='method used to partition the regressors (beckmann, guttman, ridgway, or none)')
parser.add_argument('--filterType',default='butterworth',help='type of filter to be used (butterworth or none)')
parser.add_argument('--filterOrder',default='5',help='order of high-pass filter applied to event variables')
parser.add_argument('--hpfCutoff',default='50',help='cutoff time (in sec) of high-pass filter applied to event variables (-1=no high-pass)')
parser.add_argument('--lpfCutoff',default='-1',help='cutoff time (in sec) of low-pass filter applied to event variables (-1=no low-pass)')

# ==== Declare main command-line function ==== #

if __name__ == '__main__':
    
    # ===== Set up
    
    # parse inputs
    args = parser.parse_args();
    evFiles = args.evFiles;         # list of event variable files
    evNames = args.evNames;         # list of event variable names
    conFile = args.conFile;         # name of the contrast text file
    if args.afni_timing:            # what kind of files are evFiles?
        fileType = 'afni_timing'
    elif args.fsl_timing:
        fileType = 'fsl_timing'
    elif args.fsl_design:
        fileType = 'fsl_design'
    else:
        raise ValueError("Input must include either --fsl_design, --afni_timing, or --fsl_timing flag.")
    afniDir = args.afniDir;         # directory where AFNI can be found (important if AFNI timing files are used)
    runTime = float(args.runTime);  # time per run
    TR = float(args.TR);            # MRI sequence repetition time (for event sampling)
    alpha = float(args.alpha);      # desired significance level
    tCrit = float(args.tCrit);      # desired critical t value (overrides alpha)
    noiseStd = float(args.noiseStd);# estimated noise standard deviation
    outFile = args.outFile;         # text output filename
    outImagePrefix = args.outImagePrefix; # directory where images should be saved  
    pauseToReview = args.pauseToReview;   # should we wait at the end for user to review figures?
    skipHrf = args.skipHrf;               # should we skip convolving EVs with a hemodynamic response function?
    partMethod = args.partMethod.lower(); # which method should be used to partition the data?
    filterType = args.filterType.lower(); # which type of filter should be used on the EVs?
    filterOrder = int(args.filterOrder);  # filter order
    hpfCutoff = float(args.hpfCutoff);    # high-pass filter time in seconds
    lpfCutoff = float(args.lpfCutoff);    # low-pass filter time in seconds
    
    # ===== Call main function
    EvaluateTaskTiming(evFiles,evNames,conFile,fileType,afniDir,
                    runTime,TR,alpha,tCrit,noiseStd,
                    outFile,outImagePrefix,pauseToReview,
                    skipHrf, partMethod,
                    filterType,filterOrder,hpfCutoff,lpfCutoff)