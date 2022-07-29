# Python_Cardiac

########################################################

Emma Bailey, 08/11/2021

########################################################

Set of scripts used to remove the cardiac artefact from spinal electrophysiology data using a variety of methods, namely:
PCA_OBS - Principal Component Analysis Optimal Basis Sets,
ICA - Independent Components Analysis,
ICA after PCA_OBS,
SSP - Signal Space Projection with varying numbers of projectors,
CCA - Canonical Correlation Analysis

main.py is a wrapper script to run the alternative methods listed above.

Scripts contained in /Metrics are used to compute the SNR, Residual Intensity, Improved Normalised Power Spectrum Ratio
and the Coefficient of Variation. There is also a script to extract the negative amplitude of each trial in the expected 
latency of a somatosensory evoked potential.

Scripts contained in /Statistics are used to perform a reliability assessment of each method tested, as well as obtain
p-values via paired sample permutation t-tests for the SNR, Residual Intensity and Improved Normalise Power Spectrum
results.

Scripts contained in /Plotting_Code are used to generate all images, both for debugging and visualisation purposes.

All scripts run with python 3.9 and MNE 1.0.3, and https://github.com/neurophysics/meet (to run CCA) installed in 
the project environment, for an extensive list of required packages see requirements.txt
