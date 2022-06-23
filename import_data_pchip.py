# Function to interpolate based on PCHIP rather than MNE inbuilt linear option

# In this case ch_names, current_channel and savename are dummy vars - not necessary really
    PCA_OBS_kwargs = dict(
        debug_mode=False, qrs=QRSevents_m, filter_coords=fwts, sr=sampling_rate,
        savename=save_path + 'pca_chan',
        ch_names=channelNames, sub_nr=subject_id,
        condition=cond_name, current_channel=ch
    )

    # adapted for data: delay of r - peak = 0 % (data, eventtype, method)
    # Apply function should modify the data in raw in place - checked output and it is working
    raw.apply_function(PCA_OBS_tukey, picks=esg_chans, **PCA_OBS_kwargs, n_jobs=len(esg_chans))

import mne
import numpy as np
from scipy.interpolate import PchipInterpolator as pchip
import matplotlib.pyplot as plt

# NEED TO DROP ECG CHANNEL BEFORE DOING THIS
def PCHIP_interpolation(data, **kwargs):
    # Check all necessary arguments sent in
    required_kws = ["trigger_indices", "interpol_window_msec", "fs", "debug_mode"]
    assert all([kw in kwargs.keys() for kw in required_kws]), "Error. Some KWs not passed into PCA_OBS."

    # Extract all kwargs - more elegant ways to do this
    fs = kwargs['fs']
    interpol_window_msec = kwargs['interpol_window_msec']
    trigger_indices = kwargs['trigger_indices']
    debug_mode = kwargs['debug_mode']

    if debug_mode:
        plt.figure()
        # plot signal with artifact
        plot_range = [-50, 100]
        test_trial = 100
        xx = (np.arange(plot_range[0], plot_range[1])) / fs * 1000
        plt.plot(xx, data[trigger_indices[test_trial] + plot_range[0]:trigger_indices[test_trial] + plot_range[1])

    pre_window = round(interpol_window_msec[0] * fs / 1000) # in samples
    post_window = round(interpol_window_msec[1] * fs / 1000) # in samples
    intpol_window = np.ceil([pre_window, post_window]) # interpolation window

    n_samples_fit = 5 # +1  number of samples before and after cut used for interpolation fit

    x_fit_raw = [np.arange(intpol_window[0]-n_samples_fit, intpol_window[0]+1),
                 np.arange(intpol_window[1], intpol_window[1]+n_samples_fit+1)]
    x_interpol_raw = np.arange(intpol_window[0], intpol_window[1]+1)  # points to be interpolated; in pt

    for ii in np.arange(0, len(trigger_indices)):  # loop through all stimulation events
        x_fit = trigger_indices[ii] + x_fit_raw  # fit point latencies for this event
        x_interpol = trigger_indices(ii) + x_interpol_raw  # latencies for to-be-interpolated data points

        # Data is just a string of values
        y_fit = data[x_fit]  # y values to be fitted
        y_interpol = pchip(x_fit, y_fit)(x_interpol)  # perform interpolation
        data[x_fit] = y_interpol  # replace in data

        if np.mod(ii, 100) == 0:  # talk to the operator every 100th trial
            print(f'stimulation event {ii} \n')

    if debug_mode:
        # plot signal with interpolated artifact
        plt.figure()
        plt.plot(xx, data[trigger_indices[test_trial] + plot_range[0] : trigger_indices[test_trial] + plot_range[1])
        plt.title('After Correction')

    plt.show()