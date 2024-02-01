# Just want to compare the evoked responses from SSP before and after anterior rereferncing to see difference

import mne
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw

if __name__ == '__main__':
    reduced_epochs = False
    single_subject = False
    grand_average = True

    # Testing with random subjects atm
    if single_subject:
        subjects = [1, 6, 11, 15, 22, 30]
    if grand_average:
        subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    if single_subject:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/SSP_anterior_comparison_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channels = ['S23', 'L1', 'S31']
                    full_name = 'Tibial Nerve Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channels = ['S6', 'SC6', 'S14']
                    full_name = 'Median Nerve Stimulation'

                for n in np.arange(6, 7):
                    # Load epochs resulting from SSP cleaning
                    input_path = "/data/p_02569/SSP/" + subject_id + "/" + str(n) + " projections/"
                    raw = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}.fif")
                    raw_ant = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}_antRef.fif")

                    for ch in channels:
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                        evoked_ant = evoked_from_raw(raw_ant, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                        evoked_ch = evoked.pick_channels([ch])
                        evoked_ant_ch = evoked_ant.pick_channels([ch])
                        plt.figure()
                        plt.plot(evoked_ch.times, evoked_ch.data.reshape(-1)*10**6, label='Original', color='magenta')
                        plt.plot(evoked_ant_ch.times, evoked_ant_ch.data.reshape(-1)*10**6,
                                 label='Anterior Rereferencing', color='plum')
                        plt.xlim([-0.025, 0.065])
                        plt.ylabel('Amplitude (\u03BCV)')
                        plt.xlabel('Time (s)')
                        plt.legend()
                        plt.title(f'{full_name}\n'
                                  f'Subject {subject}, Channel {ch}')

                        if cond_name == 'tibial':
                            plt.axvline(x=22 / 1000, linewidth=0.7, linestyle='dashed', color='k')
                        elif cond_name == 'median':
                            plt.axvline(x=13 / 1000, linewidth=0.7, linestyle='dashed', color='k')

                        if reduced_epochs:
                            plt.savefig(f"{figure_path}SSP_{n}proj_{ch}_{cond_name}_reduced.png")
                        else:
                            plt.savefig(f"{figure_path}SSP_{n}proj_{ch}_{cond_name}.png")
                        plt.close()

        plt.show()

    if grand_average:
        for n in np.arange(6, 7):  # Methods Applied
            figure_path = "/data/p_02569/SSP_anterior_comparison_images/"
            os.makedirs(figure_path, exist_ok=True)

            for cond_name in cond_names:  # Conditions (median, tibial)

                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channels = ['S23', 'L1', 'S32']
                    full_name = 'Tibial Nerve Stimulation'

                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channels = ['S6', 'SC6', 'S14']
                    full_name = 'Median Nerve Stimulation'

                for channel in channels:
                    evoked_list = []
                    evoked_ant_list = []

                    for subject in subjects:  # All subjects
                        subject_id = f'sub-{str(subject).zfill(3)}'
                        input_path = f"/data/p_02569/SSP/{subject_id}/{n} projections/"
                        raw = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                        raw_ant = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}_antRef.fif", preload=True)

                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                        evoked_ant = evoked_from_raw(raw_ant, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                        evoked_ch = evoked.pick_channels([channel], ordered=True)
                        evoked_ant_ch = evoked_ant.pick_channels([channel], ordered=True)

                        evoked_list.append(evoked_ch)
                        evoked_ant_list.append(evoked_ant_ch)

                    relevant_channel = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
                    relevant_ant_channel = mne.grand_average(evoked_ant_list, interpolate_bads=False, drop_bads=False)
                    # relevant_channel = averaged.pick_channels(channel)
                    plt.figure()
                    plt.plot(relevant_channel.times, np.mean(relevant_channel.data[:, :], axis=0) * 10 ** 6,
                             label='Original', color='magenta')
                    plt.plot(relevant_ant_channel.times, np.mean(relevant_ant_channel.data[:, :], axis=0) * 10 ** 6,
                             label='Anterior Rereferenced', color='plum')
                    plt.ylabel('Amplitude (\u03BCV)')
                    plt.xlabel('Time (s)')
                    plt.xlim([-0.025, 0.065])
                    if cond_name == 'tibial':
                        plt.axvline(x=22 / 1000, color='k', linewidth=0.7, linestyle='dashed', label=None)
                        # plt.ylim([-0.5, 1.3])
                    elif cond_name == 'median':
                        plt.axvline(x=13 / 1000, color='k', linewidth=0.7, linestyle='dashed', label=None)
                        # plt.ylim([-0.8, 0.8])
                    # plt.title(f"{full_name}\n"
                    #           f"Evoked Grand Average, Channel: {channel}")
                    plt.title(f"Channel: {channel}")
                    if reduced_epochs:
                        fname = f"SSP_{n}_{trigger_name}_{channel}_reducedtrials_newleg.png"
                    else:
                        fname = f"SSP_{n}_{trigger_name}_{channel}_newleg.png"
                    if channel == 'S32':
                        plt.legend(loc='upper right')
                    plt.savefig(figure_path + fname)
                    plt.clf()
