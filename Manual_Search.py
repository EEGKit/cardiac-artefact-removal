# Strange results obtained from SNR for evoked potential and heart artefact reduction
# This file is just to conduct manual searches to see if visual inspection gives any clues as to why

# Script to plot the evoked response after the different methods of heart artefact removal
# Looking at qrs events
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
    Prepared_flag = True
    PCA_flag = True
    ICA_flag = False
    post_ICA_flag = False
    SSP_flag = False
    reduced_epochs = False

    # Testing with random subjects atm
    subjects = [1, 11, 22, 30]
    # subjects = np.arange(1, 2)  # (1, 37) # 1 through 36 to access subject data
    cond_names = ['tibial', 'median']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    if Prepared_flag:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/Evoked_heart_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'qrs'
                    channels = ['S23', 'L1', 'S31']
                elif cond_name == 'median':
                    trigger_name = 'qrs'
                    channels = ['S6', 'SC6', 'S14']

                # Load epochs resulting from PCA OBS cleaning - the raw data in this folder has not been rereferenced
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr1000_{cond_name}_withqrs.fif", preload=True)
                # add reference channel to data
                mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                raw.set_eeg_reference(ref_channels='average')  # Perform rereferencing

                cfg = loadmat(cfg_path + 'cfg.mat')
                notch_freq = cfg['notch_freq'][0]
                esg_bp_freq = cfg['esg_bp_freq'][0]
                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                for ch in channels:
                    fig = evoked.plot(picks=[ch], exclude='bads', unit=True, show=False,
                                      xlim=tuple([-0.2, 0.2]), proj=True)
                    plt.title(ch)

                    # plt.show()
                    if reduced_epochs:
                        plt.savefig(f"{figure_path}Prepared_{ch}_{cond_name}_reduced.jpg")
                    else:
                        plt.savefig(f"{figure_path}Prepared_{ch}_{cond_name}.jpg")

                    plt.close()

    if PCA_flag:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/Evoked_heart_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'qrs'
                    channels = ['S23', 'L1', 'S31']
                elif cond_name == 'median':
                    trigger_name = 'qrs'
                    channels = ['S6', 'SC6', 'S14']

                # Load epochs resulting from PCA OBS cleaning - the raw data in this folder has not been rereferenced
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                raw = mne.io.read_raw_fif(f"{input_path}data_clean_ecg_spinal_{cond_name}_withqrs.fif", preload=True)
                # add reference channel to data
                mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                raw.set_eeg_reference(ref_channels='average')  # Perform rereferencing

                cfg = loadmat(cfg_path + 'cfg.mat')
                notch_freq = cfg['notch_freq'][0]
                esg_bp_freq = cfg['esg_bp_freq'][0]
                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                for ch in channels:
                    fig = evoked.plot(picks=[ch], exclude='bads', unit=True, show=False,
                                      xlim=tuple([-0.2, 0.2]), proj=True)
                    plt.title(ch)

                    # plt.show()
                    if reduced_epochs:
                        plt.savefig(f"{figure_path}PCA_{ch}_{cond_name}_reduced.jpg")
                    else:
                        plt.savefig(f"{figure_path}PCA_{ch}_{cond_name}.jpg")

                    plt.close()

    if ICA_flag:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/Evoked_heart_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'qrs'
                    channels = ['S23', 'L1', 'S31']
                elif cond_name == 'median':
                    trigger_name = 'qrs'
                    channels = ['S6', 'SC6', 'S14']

                # Load epochs resulting from baseline ICA cleaning
                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                raw = mne.io.read_raw_fif(f"{input_path}clean_baseline_ica_auto_{cond_name}.fif")

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                for ch in channels:
                    fig = evoked.plot(picks=[ch], exclude='bads', unit=True, show=False,
                                      xlim=tuple([-0.2, 0.2]), proj=True)
                    plt.title(ch)

                    # plt.show()
                    if reduced_epochs:
                        plt.savefig(f"{figure_path}ICA_{ch}_{cond_name}_reduced.jpg")
                    else:
                        plt.savefig(f"{figure_path}ICA_{ch}_{cond_name}.jpg")
                    plt.close()

    if post_ICA_flag:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/Evoked_heart_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'qrs'
                    channels = ['S23', 'L1', 'S31']
                elif cond_name == 'median':
                    trigger_name = 'qrs'
                    channels = ['S6', 'SC6', 'S14']

                input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
                raw = mne.io.read_raw_fif(f"{input_path}clean_ica_auto_{cond_name}.fif")

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                for ch in channels:
                    fig = evoked.plot(picks=[ch], exclude='bads', unit=True, show=False,
                                      xlim=tuple([-0.2, 0.2]), proj=True)
                    plt.title(ch)

                    # plt.show()
                    if reduced_epochs:
                        plt.savefig(f"{figure_path}post_ICA_{ch}_{cond_name}_reduced.jpg")
                    else:
                        plt.savefig(f"{figure_path}post_ICA_{ch}_{cond_name}.jpg")
                    plt.close()

    if SSP_flag:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/Evoked_heart_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'qrs'
                    channels = ['S23', 'L1', 'S31']
                elif cond_name == 'median':
                    trigger_name = 'qrs'
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

                        # plt.show()
                        if reduced_epochs:
                            plt.savefig(f"{figure_path}SSP_{n}proj_{ch}_{cond_name}_reduced.jpg")
                        else:
                            plt.savefig(f"{figure_path}SSP_{n}proj_{ch}_{cond_name}.jpg")
                        plt.close()
