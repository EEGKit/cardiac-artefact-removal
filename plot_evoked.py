# Script to plot the evoked response after the different methods of heart artefact removal
# Median channels to plot: ['S6', 'SC6', 'S14']
# Tibial channels to plot: ['S23', 'L1', 'S31']
# Max y = 1, min y = -1
# Only plot epochs up to 200ms for clearer visualisation

import mne
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io import loadmat
from SNR_functions import evoked_from_raw

if __name__ == '__main__':
    PCA_flag = True
    ICA_flag = True
    post_ICA_flag = True
    SSP_flag = True
    reduced_epochs = True

    # Testing with random subjects atm
    subjects = [1, 6, 11, 15, 22, 30]
    # subjects = np.arange(1, 2)  # (1, 37) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    if PCA_flag:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/Evoked_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channels = ['S23', 'L1', 'S31']
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channels = ['S6', 'SC6', 'S14']

                # Load epochs resulting from PCA OBS cleaning - the raw data in this folder has not been rereferenced
                # which is why I instead load the epoched data
                input_path = "/data/pt_02569/tmp_data/epoched_py/" + subject_id + "/esg/prepro/"
                epochs = mne.read_epochs(f"{input_path}epo_clean_{cond_name}.fif")

                if reduced_epochs:
                    epochs = epochs[900:1100]

                evoked = epochs.average()

                for ch in channels:
                    fig = evoked.plot(picks=[ch], exclude='bads', unit=True, show=False,
                                      xlim=tuple([-0.2, 0.2]), proj=True)
                    plt.title(ch)

                    if cond_name == 'tibial':
                        plt.axvline(x=22/1000, linewidth=0.5, linestyle='--')
                    elif cond_name == 'median':
                        plt.axvline(x=13/1000, linewidth=0.5, linestyle='--')

                    # plt.show()
                    if reduced_epochs:
                        plt.savefig(f"{figure_path}PCA_{ch}_{cond_name}_reduced.jpg")
                    else:
                        plt.savefig(f"{figure_path}PCA_{ch}_{cond_name}.jpg")

                    plt.close()

    if ICA_flag:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/Evoked_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channels = ['S23', 'L1', 'S31']
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channels = ['S6', 'SC6', 'S14']

                # Load epochs resulting from ICA cleaning
                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                raw = mne.io.read_raw_fif(f"{input_path}clean_baseline_ica_auto_{cond_name}.fif")

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                for ch in channels:
                    fig = evoked.plot(picks=[ch], exclude='bads', unit=True, show=False,
                                      xlim=tuple([-0.2, 0.2]), proj=True)
                    plt.title(ch)

                    if cond_name == 'tibial':
                        plt.axvline(x=22 / 1000, linewidth=0.5, linestyle='--')
                    elif cond_name == 'median':
                        plt.axvline(x=13 / 1000, linewidth=0.5, linestyle='--')

                    # plt.show()
                    if reduced_epochs:
                        plt.savefig(f"{figure_path}ICA_{ch}_{cond_name}_reduced.jpg")
                    else:
                        plt.savefig(f"{figure_path}ICA_{ch}_{cond_name}.jpg")
                    plt.close()

    if post_ICA_flag:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/Evoked_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channels = ['S23', 'L1', 'S31']
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channels = ['S6', 'SC6', 'S14']

                # Load epochs resulting from post ICA cleaning
                input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
                raw = mne.io.read_raw_fif(f"{input_path}clean_ica_auto_{cond_name}.fif")

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                for ch in channels:
                    fig = evoked.plot(picks=[ch], exclude='bads', unit=True, show=False,
                                      xlim=tuple([-0.2, 0.2]), proj=True)
                    plt.title(ch)

                    if cond_name == 'tibial':
                        plt.axvline(x=22 / 1000, linewidth=0.5, linestyle='--')
                    elif cond_name == 'median':
                        plt.axvline(x=13 / 1000, linewidth=0.5, linestyle='--')

                    # plt.show()
                    if reduced_epochs:
                        plt.savefig(f"{figure_path}post_ICA_{ch}_{cond_name}_reduced.jpg")
                    else:
                        plt.savefig(f"{figure_path}post_ICA_{ch}_{cond_name}.jpg")
                    plt.close()

    if SSP_flag:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/Evoked_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channels = ['S23', 'L1', 'S31']
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channels = ['S6', 'SC6', 'S14']

                for n in np.arange(5, 21):
                    # Load epochs resulting from SSP cleaning
                    input_path = "/data/p_02569/SSP/" + subject_id + "/" + str(n) + " projections/"
                    raw = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}.fif")

                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                    for ch in channels:
                        fig = evoked.plot(picks=[ch], exclude='bads', unit=True, show=False,
                                          xlim=tuple([-0.2, 0.2]), proj=True)
                        plt.title(ch)

                        if cond_name == 'tibial':
                            plt.axvline(x=22 / 1000, linewidth=0.5, linestyle='--')
                        elif cond_name == 'median':
                            plt.axvline(x=13 / 1000, linewidth=0.5, linestyle='--')

                        # plt.show()
                        if reduced_epochs:
                            plt.savefig(f"{figure_path}SSP_{n}proj_{ch}_{cond_name}_reduced.jpg")
                        else:
                            plt.savefig(f"{figure_path}SSP_{n}proj_{ch}_{cond_name}.jpg")
                        plt.close()
