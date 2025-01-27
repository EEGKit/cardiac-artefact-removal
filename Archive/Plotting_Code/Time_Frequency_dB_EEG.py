# Script to plot the time-frequency decomposition in dB about the heartbeat
# Can't do this for CCA corrected data as it is already epoched about the spinal triggers
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets

import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt

if __name__ == '__main__':
    cortical_channels = True  # Calculate TFR for cortical channels of interest
    ECG_channel = False  # Calculate TFR for ECG channel
    freqs = np.arange(5., 250., 3.)
    fmin, fmax = freqs[[0, -1]]

    subjects = np.arange(1, 37)
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    # Want 200ms before R-peak and 400ms after R-peak
    # Baseline is the 100ms period before the artefact occurs
    iv_baseline = [-300 / 1000, -200 / 1000]
    # Want 200ms before and 400ms after the R-peak in our epoch - need baseline outside this
    iv_epoch = [-300 / 1000, 400 / 1000]

    image_path = "/data/p_02569/Images/TimeFrequencyPlots_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    # For the EEG channels of interest
    if cortical_channels:
        for cond_name in cond_names:  # Conditions (median, tibial)
            evoked_list = []

            if cond_name == 'tibial':
                trigger_name = 'qrs'
                channel = ['Cz']
                full_name = 'Tibial Nerve Stimulation'

            elif cond_name == 'median':
                trigger_name = 'qrs'
                channel = ['CP4']
                full_name = 'Median Nerve Stimulation'

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
                raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif", preload=True)
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
            im = ax.images
            cb = im[-1].colorbar
            cb.set_label('Amplitude [dB]')

            plt.title(f"Channel: {channel[0]}\n"
                      f"{full_name}")
            fname = f"{channel[0]}_{trigger_name}_dB.png"
            plt.savefig(image_path + fname)
            plt.clf()

    # For the ECG channels of interest
    if ECG_channel:
        for cond_name in cond_names:  # Conditions (median, tibial)
            evoked_list = []

            if cond_name == 'tibial':
                trigger_name = 'qrs'
                channel = ['ECG']
                full_name = 'Tibial Nerve Stimulation'

            elif cond_name == 'median':
                trigger_name = 'qrs'
                channel = ['ECG']
                full_name = 'Median Nerve Stimulation'

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
                raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif",
                                          preload=True)
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked = evoked.pick_channels(channel)
                evoked = evoked.set_channel_types({'ECG': 'eeg'})  # Needed to do transform
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
            im = ax.images
            cb = im[-1].colorbar
            cb.set_label('Amplitude [dB]')

            plt.title(f"Channel: {channel[0]}\n"
                      f"{full_name}")
            fname = f"{channel[0]}_{trigger_name}_{cond_name}_dB.png"
            plt.savefig(image_path + fname)
            plt.clf()
