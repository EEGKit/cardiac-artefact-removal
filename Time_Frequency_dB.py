# Script to plot the time-frequency decomposition in dB about the heartbeat
# Can't do this for CCA corrected data as it is already epoched about the spinal triggers
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets

import mne
import os
import numpy as np
from scipy.io import loadmat
from SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt
from get_esg_channels import get_esg_channels

if __name__ == '__main__':
    low_freq_flag = True
    if low_freq_flag:
        freqs = np.arange(5., 250., 3.)
        fmin, fmax = freqs[[0, -1]]
    else:
        freqs = np.arange(0., 250., 3.)
        fmin, fmax = freqs[[0, -1]]

    spinal = False  # If true plots SEPs, if false plots QRS
    subjects = np.arange(1, 37)
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    if spinal:
        iv_epoch = cfg['iv_epoch'][0] / 1000
        iv_baseline = cfg['iv_baseline'][0] / 1000
    else:
        iv_baseline = [-150 / 1000, -50 / 1000]
        iv_epoch = [-200 / 1000, 200 / 1000]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/TimeFrequencyPlots_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    methods = [True, True, True, True]  # No need to do ICA - can't perform CCA on data
    method_names = ['Prep', 'PCA', 'ICA', 'Post-ICA']  # Will treat SSP separately since there are multiple
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
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                        raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif", preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel)
                        power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                        # evoked_list.append(evoked)
                        evoked_list.append(power)

                    elif method == 'PCA':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel)
                        power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                        # evoked_list.append(evoked)
                        evoked_list.append(power)

                    elif method == 'ICA':
                        input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                        fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel)
                        power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                        # evoked_list.append(evoked)
                        evoked_list.append(power)

                    elif method == 'Post-ICA':
                        input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
                        fname = f"clean_ica_auto_{cond_name}.fif"
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
                    if method == 'Prep':
                        vmin=-400
                        vmax=-175
                    else:
                        if cond_name == 'tibial':
                            vmin=-400
                            vmax=-225
                        else:
                            vmin=-400
                            vmax=-250
                    tmin = -0.2
                    tmax = 0.2
                fig, ax = plt.subplots(1, 1)
                # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                if low_freq_flag:
                    averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                                  axes=ax, show=False, colorbar=True, dB=True, fmin=0, fmax=20,
                                  tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
                else:
                    averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                                  axes=ax, show=False, colorbar=True, dB=True,
                                  tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)

                plt.title(f"Method: {method}, Condition: {trigger_name}")
                if spinal:
                    fname = f"{method}_{trigger_name}_{cond_name}_dB.png"
                else:
                    if low_freq_flag:
                        fname = f"{method}_{trigger_name}_{cond_name}_dB_low.png"
                    else:
                        fname = f"{method}_{trigger_name}_{cond_name}_dB.png"
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

                    input_path = f"/data/p_02569/SSP/{subject_id}/{n} projections/"
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
                else:
                    tmin = -0.2
                    tmax = 0.2
                    if cond_name == 'tibial':
                        vmin = -400
                        vmax = -225
                    else:
                        vmin = -400
                        vmax = -250
                fig, ax = plt.subplots(1, 1)
                # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.1)
                if low_freq_flag:
                    averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                                  axes=ax, show=False, colorbar=True, dB=True, fmin=0, fmax=20,
                                  tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
                else:
                    averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                                  axes=ax, show=False, colorbar=True, dB=True,
                                  tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)

                plt.title(f"Method: {method}, Condition: {trigger_name}")
                if spinal:
                    fname = f"{method}_{trigger_name}_{cond_name}_dB.png"
                else:
                    if low_freq_flag:
                        fname = f"{method}_{trigger_name}_{cond_name}_dB_low.png"
                    else:
                        fname = f"{method}_{trigger_name}_{cond_name}_dB.png"

                plt.savefig(image_path + fname)
                plt.clf()
