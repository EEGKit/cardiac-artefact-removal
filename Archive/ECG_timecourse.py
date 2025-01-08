# Code to plot grand average time course of the ECG channel


import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt

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

    image_path = "/data/p_02569/Images/ECGGrandAverage_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    # For the ECG channels of interest
    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list = []

        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channel = ['ECG']

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channel = ['ECG']

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
            raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif",
                                      preload=True)
            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked = evoked.pick_channels(channel)
            evoked_list.append(evoked)

        relevant_channel = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
        # relevant_channel = averaged.pick_channels(channel)
        plt.plot(relevant_channel.times, np.mean(relevant_channel.data[:, :], axis=0) * 10 ** 6,
                 label='Evoked Grand Average')
        plt.ylabel('Amplitude [\u03BCV]')

        plt.xlabel('Time [s]')
        plt.xlim([-0.2, 0.2])
        plt.title(f"ECG Grand Average")
        fname = f"ECG_GrandAverage_{cond_name}.png"
        plt.savefig(image_path + fname)
        plt.clf()
