# EvaluateTaskTiming
Evaluate a given set of event timings and contrasts for an fMRI task, using several metrics based on regressor independence and efficiency.

Reads timing files (.1D) from AFNI or design (.mat) or timing (3-column .txt) 
files from FSL and evaluates them using 4 different metrics: 
1) Mean square off-diagonal correlation between event regressors
2) Flatness of scree plot (normalized SVD eigenvalues of event regressor matrix)
3) Experimental Design Efficiency (based on Henson, 2007, eq. 15.4)
4) BOLD Signal Change Required for Significance at specified alpha value 
(based on Smith, 2007, eq. (5)
Outputs the results as figures and a single-row text file for comparison across
task designs.

See python EvaluateTaskTiming.py -h for command-line usage and input information, or import and use the EvaluateTaskTiming function as in RunSingleBoxcarTest.py.

Samples of all three timing file types can be found in the SampleTimingFiles folder.

References:
* Henson, R. (2007). Efficient experimental design for fMRI. Statistical 
parametric mapping: The analysis of functional brain images, 193-210.
* Smith SM, Jenkinson M, Beckmann C, Miller K,
Woolrich M. Meaningful design and contrast estimability
in FMRI. Neuroimage 2007;34(1):127-36.
