import numpy as np
import mne
from scipy.signal import filtfilt, detrend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from fit_ecgTemplate import fit_ecgTemplate
import math
import h5py

def PCA_OBS_fittedart(data, **kwargs):

    # Declare class to hold pca information
    class PCAInfo():
        def __init__(self):
            pass

    # Instantiate class
    pca_info = PCAInfo()

    # Check all necessary arguments sent in
    required_kws = ["debug_mode", "qrs", "sr", "savename", "ch_names", "sub_nr", "condition", "current_channel"]
    assert all([kw in kwargs.keys() for kw in required_kws]), "Error. Some KWs not passed into PCA_OBS."

   # Extract all kwargs - more elegant ways to do this
    debug_mode = kwargs['debug_mode']
    qrs = kwargs['qrs']
    sr = kwargs['sr']
    ch_names = kwargs['ch_names']
    sub_nr = kwargs['sub_nr']
    condition = kwargs['condition']
    if debug_mode:  # Only need current channel and saving if we're debugging
        current_channel = kwargs['current_channel']  # This is the awkward one for parallelisation
        savename = kwargs['savename']

    fs = sr

    # Standard delay between QRS peak and artifact
    delay = 0

    Gwindow = 2 # 20 adapted by BN 13.05.2019
    GHW = np.floor(Gwindow / 2).astype('int')
    rcount = 0
    firstplot = 1

    # set to baseline - editing shape to match MATLAB code (data there is 40 channels x 1567442 times)
    # whereas here the data extracted is showing as (1567442, )
    data = data.reshape(-1, 1)
    data = data.T
    data = data - np.mean(data, axis=1) # axis = 1 to compute mean of each row

    # Allocate memory
    fitted_art = np.zeros(data.shape)
    peakplot = np.zeros(data.shape)

    # Extract QRS events
    # If statement to avoid indexing out of bounds - problem first noticed with subject 31
    # qrs is a list of lists
    # Original code: peakplot[0, qrs] = 1
    for idx in qrs[0]:
        if idx < len(peakplot[0, :]):
            peakplot[0, idx] = 1 # logical indexed locations of qrs events
    # sh = np.zeros((1, delay))
    # np1 = len(peakplot)
    # peakplot = [sh, peakplot(1:np1 - delay)] # shifts indexed array by the delay - skipped here since delay=0
    peak_idx = np.nonzero(peakplot)[1] # Selecting indices along columns
    peak_idx = peak_idx.reshape(-1, 1)
    peak_count = len(peak_idx)

    ################################################################
    # Preparatory work - reserving memory, configure sizes, de-trend
    ################################################################
    print('Pulse artifact subtraction in progress...Please wait!')

    # define peak range based on RR
    RR = np.diff(peak_idx[:, 0]) # peak_idx is (1270, 1) - need to extract the 1270
    mRR = np.median(RR)
    peak_range = round(mRR/2) # Rounds to an integer
    midP = peak_range + 1
    baseline_range = [0, round(peak_range/8)] # [1 round((peak_range / 8) * 3)]
    n_samples_fit = round(peak_range/8) # sample fit for interpolation between fitted artifact windows

    # make sure array is long enough for PArange (if not cut off  last ECG peak)
    pa = peak_count # Number of QRS complexes detected
    # peak_idx is (1270, 1) - want to access last element in first column
    while peak_idx[pa-1, 0] + peak_range > len(data[0]):
        pa = pa - 1
    steps = 1 * pa
    peak_count = pa

    # build PCA matrix(heart - beat - epoch x window - length)
    pcamat = np.zeros((peak_count - 1, 2*peak_range+1)) # [epoch x time]
    # dpcamat = pcamat # [epoch x time]
    # picking out heartbeat epochs
    for p in range(1, peak_count):
        pcamat[p-1, :] = data[0, peak_idx[p, 0] - peak_range: peak_idx[p, 0] + peak_range+1]

    # detrending matrix(twice - why?)
    pcamat = detrend(pcamat, type='constant', axis=1) # [epoch x time] - detrended along the epoch
    # pcamat = np.transpose(detrend(np.transpose(pcamat), type='constant'))
    mean_effect = np.mean(pcamat, axis=0) # [1 x time], contains the mean over all epochs
    std_effect = np.std(pcamat, axis=0) # want mean and std of each column
    dpcamat = detrend(pcamat, type='constant', axis=1) # [time x epoch]

    ###################################################################
    # Perform PCA with sklearn
    ###################################################################
    # run PCA(performs SVD(singular value decomposition))
    # Using sklearn for this step - matlab translation was a mess
    # pca = PCA()
    pca = PCA(svd_solver="full")
    pca.fit(dpcamat)
    eigen_vectors = pca.components_
    eigen_values = pca.explained_variance_
    factor_loadings = pca.components_.T*np.sqrt(pca.explained_variance_)
    pca_info.eigen_vectors = eigen_vectors
    pca_info.factor_loadings = factor_loadings
    pca_info.eigen_values = eigen_values
    pca_info.expl_var = pca.explained_variance_ratio_

    # define selected number of  components using profile likelihood
    pca_info.nComponents = 4

    pca_info.meanEffect = mean_effect.T
    nComponents = pca_info.nComponents

    #######################################################################
    # Make template of the ECG artefact
    #######################################################################
    mean_effect = mean_effect.reshape(-1, 1)
    pca_template = np.c_[mean_effect, factor_loadings[:, 0:nComponents]]

    ###################################################################################
    # Data Fitting
    ###################################################################################
    window_start_idx = []
    window_end_idx = []
    for p in range(0, peak_count):
        # Deals with start portion of data
        if p == 0:
            pre_range = peak_range
            post_range = math.floor((peak_idx[p + 1] - peak_idx[p])/2)
            if post_range > peak_range:
                post_range = peak_range
            try:
                post_idx_nextPeak = []
                fitted_art, post_idx_nextPeak = fit_ecgTemplate(data, pca_template, peak_idx[p], peak_range,
                                                                pre_range, post_range, baseline_range, midP,
                                                                fitted_art, post_idx_nextPeak, n_samples_fit)
                # Appending to list instead of using counter
                window_start_idx.append(peak_idx[p] - peak_range)
                window_end_idx.append(peak_idx[p] + peak_range)
            except Exception as e:
                print(f'Cannot fit first ECG epoch. Reason: {e}')

        # Deals with last edge of data
        elif p == peak_count - 1:
            print('On last section - almost there!')
            try:
                pre_range = math.floor((peak_idx[p] - peak_idx[p - 1]) / 2)
                post_range = peak_range
                if pre_range > peak_range:
                    pre_range = peak_range
                fitted_art, _ = fit_ecgTemplate(data, pca_template, peak_idx(p), peak_range, pre_range, post_range,
                                                baseline_range, midP, fitted_art, post_idx_nextPeak, n_samples_fit)
                window_start_idx.append(peak_idx[p] - peak_range)
                window_end_idx.append(peak_idx[p] + peak_range)
            except Exception as e:
                print(f'Cannot fit last ECG epoch. Reason: {e}')

        # Deals with middle portion of data
        else:
            try:
                # ---------------- Processing of central data - --------------------
                # cycle through peak artifacts identified by peakplot
                pre_range = math.floor((peak_idx[p] - peak_idx[p - 1]) / 2)
                post_range = math.floor((peak_idx[p + 1] - peak_idx[p]) / 2)
                if pre_range >= peak_range:
                    pre_range = peak_range # peak_range - 2
                if post_range > peak_range:
                    post_range = peak_range

                aTemplate = pca_template[midP - peak_range-1:midP + peak_range+1, :]
                fitted_art, post_idx_nextPeak = fit_ecgTemplate(data, aTemplate, peak_idx[p], peak_range, pre_range,
                                                                post_range, baseline_range, midP, fitted_art,
                                                                post_idx_nextPeak, n_samples_fit)
                window_start_idx.append(peak_idx[p] - peak_range)
                window_end_idx.append(peak_idx[p] + peak_range)
            except Exception as e:
                print(f"Cannot fit middle section of data. Reason: {e}")

    # Actually subtract the artefact, return needs to be the same shape as input data
    # One sample shift purely due to the fact the r-peaks are currently detected in MATLAB
    data = data.reshape(-1)
    fitted_art = fitted_art.reshape(-1)

    # Can only return data
    return fitted_art