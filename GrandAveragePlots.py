# Script to create plots of the grand averages evoked responses across participants for each stimulation

import mne
import os
import numpy as np
from scipy.io import loadmat
from SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt


if __name__ == '__main__':
    subjects = np.arange(1, 25)   # 1 through 24 to access subject data
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

    image_path = "/data/p_02569/GrandAveragePlots_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    methods = [True, True, False, True]
    method_names = ['Prep', 'PCA', 'ICA', 'Post-ICA']  # Will treat SSP separately since there are multiple
    # If cca flag is True, ICA must be False
    cca_flag = True  # If you want to plot the results of the CCA, all CCA files are filtered already

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for i in np.arange(0, len(methods)):  # Methods Applied
        if methods[i]:  # Allows us to apply to only methods of interest
            method = method_names[i]

            for cond_name in cond_names:  # Conditions (median, tibial)
                evoked_list = []

                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channel = ['L1']

                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channel = ['SC6']

                if cca_flag:
                    channel = ['Cor1', 'Cor2']

                for subject in subjects:  # All subjects
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    if method == 'Prep':
                        if cca_flag:
                            input_path = "/data/pt_02569/tmp_data/prepared_py_cca/" + subject_id + "/esg/prepro/"
                            epochs = mne.read_epochs(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                                                     , preload=True)
                            evoked = epochs.average()
                            evoked_list.append(evoked)
                        else:
                            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                            raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                                                      , preload=True)
                            mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                       method='iir',
                                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                            evoked.reorder_channels(esg_chans)
                            evoked_list.append(evoked)

                    elif method == 'PCA':
                        if cca_flag:
                            input_path = "/data/pt_02569/tmp_data/ecg_rm_py_cca/" + subject_id + "/esg/prepro/"
                            fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                            epochs = mne.read_epochs(input_path + fname, preload=True)
                            evoked = epochs.average()
                            evoked_list.append(evoked)
                        else:
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
                            evoked_list.append(evoked)

                    elif method == 'ICA':
                        input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                        fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked_list.append(evoked)

                    elif method == 'Post-ICA':
                        if cca_flag:
                            input_path = "/data/pt_02569/tmp_data/ica_py_cca/" + subject_id + "/esg/prepro/"
                            fname = f"clean_ica_auto_{cond_name}.fif"
                            epochs = mne.read_epochs(input_path + fname, preload=True)
                            evoked = epochs.average()
                            evoked_list.append(evoked)
                        else:
                            input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
                            fname = f"clean_ica_auto_{cond_name}.fif"
                            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                            evoked.reorder_channels(esg_chans)
                            evoked_list.append(evoked)

                averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
                relevant_channel = averaged.pick_channels(channel)
                if cca_flag:
                    plt.plot(relevant_channel.times, np.mean(relevant_channel.data[:, :], axis=0),
                             label='Evoked Grand Average')
                    plt.ylabel('Amplitude [a.u.]')

                else:
                    plt.plot(relevant_channel.times, np.mean(relevant_channel.data[:, :], axis=0)*10**6,
                             label='Evoked Grand Average')
                    plt.ylabel('Amplitude [\u03BCV]')

                plt.xlabel('Time [s]')
                plt.xlim([-0.1, 0.3])
                if cond_name == 'tibial':
                    plt.axvline(x=22 / 1000, color='r', linewidth=0.5, label='22ms')
                elif cond_name == 'median':
                    plt.axvline(x=13 / 1000, color='r', linewidth=0.5, label='13ms')
                plt.title(f"Method: {method}, Condition: {trigger_name} CCA: {cca_flag}")
                if cca_flag:
                    fname = f"{method}_{trigger_name}_cca.png"
                else:
                    fname = f"{method}_{trigger_name}.png"
                plt.legend(loc='upper right')
                plt.savefig(image_path+fname)
                plt.clf()

    # Now deal with SSP plots - Just doing 5 to 10 for now
    for n in np.arange(5, 11):  # Methods Applied
        for cond_name in cond_names:  # Conditions (median, tibial)
            evoked_list = []

            if cond_name == 'tibial':
                trigger_name = 'Tibial - Stimulation'
                channel = 'L1'

            elif cond_name == 'median':
                trigger_name = 'Median - Stimulation'
                channel = 'SC6'

            if cca_flag:
                channel = ['Cor1', 'Cor2']

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                if cca_flag:
                    input_path = f"/data/p_02569/SSP_cca/{subject_id}/{n} projections/"
                    epochs = mne.read_epochs(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    evoked = epochs.average()
                    evoked_list.append(evoked)
                else:
                    input_path = f"/data/p_02569/SSP/{subject_id}/{n} projections/"
                    raw = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                    evoked.reorder_channels(esg_chans)
                    evoked_list.append(evoked)

            averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
            relevant_channel = averaged.pick_channels(channel)
            if cca_flag:
                plt.plot(relevant_channel.times, np.mean(relevant_channel.data[:, :], axis=0),
                         label='Evoked Grand Average')
                plt.ylabel('Amplitude [a.u.]')
            else:
                plt.plot(relevant_channel.times, np.mean(relevant_channel.data[:, :], axis=0) * 10 ** 6,
                         label='Evoked Grand Average')
                plt.ylabel('Amplitude [\u03BCV]')

            plt.xlabel('Time [s]')
            plt.xlim([-0.1, 0.3])
            if cond_name == 'tibial':
                plt.axvline(x=22 / 1000, color='r', linewidth=0.5, label='22ms')
            elif cond_name == 'median':
                plt.axvline(x=13 / 1000, color='r', linewidth=0.5, label='13ms')
            plt.title(f"Method: SSP {n} proj., Condition: {trigger_name}, CCA: {cca_flag}")
            if cca_flag:
                fname = f"SSP_{n}_{trigger_name}_cca.png"
            else:
                fname = f"SSP_{n}_{trigger_name}.png"
            plt.legend(loc='upper right')
            plt.savefig(image_path + fname)
            plt.clf()