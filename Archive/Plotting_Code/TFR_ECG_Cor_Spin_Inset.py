# Script to plot the time-frequency decomposition in dB about the heartbeat
# TFRs of the heart artefact in ECG, Spinal and Cortical Channels

import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


if __name__ == '__main__':
    freqs = np.arange(5., 250., 3.)
    fmin, fmax = freqs[[0, -1]]
    subjects = np.arange(1, 37)
    # subjects = [1]
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

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/TimeFrequencyPlots_HeartInset_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    channel_types = ['ECG', 'Spinal', 'Cortical']
    fig, (ax_ecg, ax_spinal, ax_cortical) = plt.subplots(1, 3, figsize=(20, 6))

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for channel_type in channel_types:
        for cond_name in cond_names:  # Conditions (median, tibial)
            evoked_list = []

            if cond_name == 'tibial':
                full_name = 'Tibial Nerve Stimulation'
                trigger_name = 'qrs'
                if channel_type == 'ECG':
                    channel = ['ECG']
                    ax = ax_ecg
                elif channel_type == 'Spinal':
                    channel = ['L1']
                    ax = ax_spinal
                elif channel_type == 'Cortical':
                    channel = ['Cz']
                    ax = ax_cortical

            elif cond_name == 'median':
                full_name = 'Median Nerve Stimulation'
                trigger_name = 'qrs'
                if channel_type == 'ECG':
                    channel = ['ECG']
                    ax = inset_axes(ax_ecg, width='40%', height='35%', loc=1, borderpad=1)
                elif channel_type == 'Spinal':
                    channel = ['SC6']
                    ax = inset_axes(ax_spinal, width='40%', height='35%', loc=1, borderpad=1)
                elif channel_type == 'Cortical':
                    channel = ['CP4']
                    ax = inset_axes(ax_cortical, width='40%', height='35%', loc=1, borderpad=1)

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                if channel_type == 'ECG' or channel_type == 'Cortical':
                    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
                    raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif",
                                              preload=True)
                elif channel_type == 'Spinal':
                    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
                    raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif",
                                              preload=True)

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked = evoked.pick_channels(channel)
                if channel_type == 'ECG':
                    evoked = evoked.set_channel_types({'ECG': 'eeg'})  # Needed to do transform
                power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                # evoked_list.append(evoked)
                evoked_list.append(power)

            averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
            # relevant_channel = averaged.pick_channels(channel)

            tmin = -0.2
            tmax = 0.4
            vmin = -400
            vmax = -175

            # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                          axes=ax, show=False, colorbar=False, dB=True,
                          tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
            if cond_name == 'tibial':
                if channel_type == 'ECG':
                    ax.set_title('ECG Channel')
                elif channel_type == 'Spinal':
                    ax.set_title('Spinal Channel')
                elif channel_type == 'Cortical':
                    ax.set_title('Cortical Channel')

            if cond_name == 'median':
                # ax.set_xticks([])
                # ax.set_yticks([])
                ax.set_xlabel(None)
                ax.set_ylabel(None)

    # Add colorbar
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.2, hspace=0.02)
    cbar_ax = fig.add_axes([0.83, 0.1, 0.01, 0.8])
    cb = fig.colorbar(ax_ecg.images[-1], cax=cbar_ax)
    # cb = im[-1].colorbar
    cb.set_label('Amplitude (dB)')
    fname = f"CardiacTFRInset_dB.png"
    fig.savefig(image_path+fname)
    plt.clf()
