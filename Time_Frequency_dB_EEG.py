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
    freqs = np.arange(5., 250., 3.)
    fmin, fmax = freqs[[0, -1]]

    subjects = np.arange(1, 37)
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    iv_baseline = [-150 / 1000, -50 / 1000]
    iv_epoch = [-200 / 1000, 200 / 1000]

    image_path = "/data/p_02569/TimeFrequencyPlots_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    # For the EEG channels of interest
    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list = []

        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channel = ['Cz']

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channel = ['CP4']

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
            raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif", preload=True)
            mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
            raw.set_eeg_reference(ref_channels='average')  # Perform rereferencing
            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                       method='iir',
                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked = evoked.pick_channels(channel)
            power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            # evoked_list.append(evoked)
            evoked_list.append(power)

        averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
        tmin = -0.2
        tmax = 0.2
        vmin = -400
        vmax = -175
        fig, ax = plt.subplots(1, 1)
        # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
        averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                      axes=ax, show=False, colorbar=True, dB=True,
                      tmin=tmin, tmax=tmax, vmin=-400, vmax=-175)

        plt.title(f"Channel: {channel[0]}, Condition: {trigger_name}")
        fname = f"{channel[0]}_{trigger_name}_dB.png"
        plt.savefig(image_path + fname)
        plt.clf()
