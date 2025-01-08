# Checking PCA single subject plots to see who drives artefact at stimulation time

import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plot_time = True
    plot_image = False
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/Images/SingleSubjectPlots_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    if plot_time:
        for cond_name in cond_names:  # Conditions (median, tibial)

            # Changing to pick channel when each is loaded, not after the evoked list is formed
            if cond_name == 'tibial':
                trigger_name = 'Tibial - Stimulation'
                channel = ['L1']

            elif cond_name == 'median':
                trigger_name = 'Median - Stimulation'
                channel = ['SC6']

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked.reorder_channels(esg_chans)
                evoked = evoked.pick_channels(channel, ordered=True)

                # relevant_channel = averaged.pick_channels(channel)
                plt.plot(evoked.times, np.mean(evoked.data[:, :], axis=0)*10**6,
                         label='Evoked Grand Average')
                plt.ylabel('Amplitude [\u03BCV]')

                plt.xlabel('Time [s]')
                plt.xlim([-0.1, 0.3])
                if cond_name == 'tibial':
                    plt.axvline(x=22 / 1000, color='r', linewidth=0.5, label='22ms')
                elif cond_name == 'median':
                    plt.axvline(x=13 / 1000, color='r', linewidth=0.5, label='13ms')
                plt.title(f"Subject {subject_id}, Method: PCA, Condition: {trigger_name}")
                fname = f"PCA_{subject}_{trigger_name}.png"
                plt.legend(loc='upper right')
                plt.savefig(image_path+fname)
                plt.clf()

    if plot_image:
        for cond_name in cond_names:  # Conditions (median, tibial)

            # Changing to pick channel when each is loaded, not after the evoked list is formed
            if cond_name == 'tibial':
                trigger_name = 'Tibial - Stimulation'
                channel = ['L1']

            elif cond_name == 'median':
                trigger_name = 'Median - Stimulation'
                channel = ['SC6']
                continue

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                    baseline=tuple(iv_baseline), preload=True)
                epochs = epochs.crop(tmin=-0.1, tmax=0.3)

                # relevant_channel = averaged.pick_channels(channel)
                epochs.plot_image(picks=channel, title=f"Subject {subject_id}, Method: PCA, Condition: {trigger_name}",
                                  vmin=-5, vmax=5, show=False)
                fname = f"PCA_{subject}_{trigger_name}_epo.png"
                plt.legend(loc='upper right')
                plt.savefig(image_path + fname)
                plt.clf()
