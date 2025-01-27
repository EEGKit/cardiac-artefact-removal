# Script to plot the time-frequency decomposition about the heartbeat
# Uses the grand averaged evoked response across all participants
# Can't do this for CCA corrected data as it is already epoched about the spinal triggers
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets

import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt

if __name__ == '__main__':
    freqs = np.arange(5., 250., 3.)
    fmin, fmax = freqs[[0, -1]]

    spinal = False  # If true plots SEPs, if false plots QRS
    subjects = np.arange(1, 37)
    cond_names = ['median', 'tibial']
    # cond_names = ['tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    if spinal:
        iv_epoch = cfg['iv_epoch'][0] / 1000
        iv_baseline = cfg['iv_baseline'][0] / 1000
    else:
        # Want 200ms before R-peak and 400ms after R-peak
        # Baseline is the 100ms period before the artefact occurs
        iv_baseline = [-300 / 1000, -200 / 1000]
        # Want 200ms before and 400ms after the R-peak in our epoch - need baseline outside this
        iv_epoch = [-300 / 1000, 400 / 1000]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/Images/TimeFrequencyPlots_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    methods = [True, True, True]  # No need to do ICA - can't perform CCA on data
    method_names = ['Prep', 'PCA', 'ICA']  # Will treat SSP separately since there are multiple
    ssp = True  # Using files from merging mixed nerve with digits

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for i in np.arange(0, len(methods)):  # Methods Applied
        if methods[i]:  # Allows us to apply to only methods of interest
            method = method_names[i]

            for cond_name in cond_names:  # Conditions (median, tibial)
                evoked_list = []

                if cond_name == 'tibial':
                    if spinal:
                        trigger_name = 'Tibial - Stimulation'
                    else:
                        trigger_name = 'qrs'
                    channel = ['L1']

                elif cond_name == 'median':
                    if spinal:
                        trigger_name = 'Median - Stimulation'
                    else:
                        trigger_name = 'qrs'
                    channel = ['SC6']

                for subject in subjects:  # All subjects
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    if method == 'Prep':
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
                        raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif", preload=True)
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel)
                        power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                        # evoked_list.append(evoked)
                        evoked_list.append(power)

                    elif method == 'PCA':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id
                        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel)
                        power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                        # evoked_list.append(evoked)
                        evoked_list.append(power)

                    elif method == 'ICA':
                        input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id
                        fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel)
                        power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                        # evoked_list.append(evoked)
                        evoked_list.append(power)

                averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
                # relevant_channel = averaged.pick_channels(channel)

                if spinal:
                    tmin = -0.1
                    tmax = 0.1
                else:  # Letting it autoset the limits for the spinal triggers - too hard to guess
                    tmin = -0.2
                    tmax = 0.2
                    if method == 'Prep':
                        vmin = 0  # -4*10**-10
                        vmax = 4*10**-10
                    elif method == 'PCA':
                        if cond_name == 'median':
                            vmin = 0  # -4 * 10 ** -14
                            vmax = 4 * 10 ** -14
                        else:
                            vmin = 0 #-8 * 10 ** -13
                            vmax = 8 * 10 ** -13
                    elif method == 'ICA':
                        if cond_name == 'median':
                            vmin = 0  # -6 * 10 ** -15
                            vmax = 6 * 10 ** -15
                        else:
                            vmin = 0  # -6 * 10 ** -12
                            vmax = 6 * 10 ** -12
                fig, ax = plt.subplots(1, 1)
                # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                if spinal:
                    averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                               axes=ax, show=False, colorbar=True,
                               tmin=tmin, tmax=tmax)
                else:
                    averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                               axes=ax, show=False, colorbar=True, vmin=vmin, vmax=vmax,
                               tmin=tmin, tmax=tmax)

                plt.title(f"Method: {method}, Condition: {trigger_name}")
                if spinal:
                    fname = f"{method}_{trigger_name}_{cond_name}.png"
                else:
                    fname = f"{method}_{trigger_name}_{cond_name}_from0.png"

                plt.savefig(image_path+fname)
                plt.clf()

    # Now deal with SSP plots - Just doing 5 to 6 for now
    if ssp:  # Actually using segregated ssp again
        for n in np.arange(5, 7):  # Methods Applied
            method = f'SSP_{n}'
            for cond_name in cond_names:  # Conditions (median, tibial)
                evoked_list = []

                if cond_name == 'tibial':
                    if spinal:
                        trigger_name = 'Tibial - Stimulation'
                    else:
                        trigger_name = 'qrs'
                    channel = ['L1']

                elif cond_name == 'median':
                    if spinal:
                        trigger_name = 'Median - Stimulation'
                    else:
                        trigger_name = 'qrs'
                    channel = ['SC6']

                for subject in subjects:  # All subjects
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    input_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/{n} projections/"
                    raw = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                    evoked.reorder_channels(esg_chans)
                    evoked = evoked.pick_channels(channel)
                    power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                    # evoked_list.append(evoked)
                    evoked_list.append(power)

                averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
                # relevant_channel = averaged.pick_channels(channel)
                if spinal:
                    tmin = -0.1
                    tmax = 0.1
                else:  # Letting it autoset the limits for the spinal triggers - too hard to guess
                    vmin = 0  # -1.5 * 10 ** -14
                    vmax = 1.5 * 10 ** -14
                    tmin = -0.2
                    tmax = 0.2
                fig, ax = plt.subplots(1, 1)
                # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.1)
                if spinal:
                    averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                               axes=ax, show=False, colorbar=True,
                               tmin=tmin, tmax=tmax)
                else:
                    averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                               axes=ax, show=False, colorbar=True, vmin=vmin, vmax=vmax,
                               tmin=tmin, tmax=tmax)

                plt.title(f"Method: {method}, Condition: {trigger_name}")
                if spinal:
                    fname = f"{method}_{trigger_name}_{cond_name}.png"
                else:
                    fname = f"{method}_{trigger_name}_{cond_name}_from0.png"

                plt.savefig(image_path + fname)
                plt.clf()
