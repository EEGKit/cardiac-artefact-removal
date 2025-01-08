# Script to create plots of the grand averages evoked responses across participants for each stimulation
# For CCA data now
# Can't use mne grand average as there are different channels involved for different subjects

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from invert import invert

if __name__ == '__main__':
    subjects = np.arange(1, 37)   # 1 through 24 to access subject data
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

    methods = [True, True]
    method_names = ['Prep', 'PCA']  # Will treat SSP separately since there are multiple

    for i in np.arange(0, len(methods)):  # Methods Applied
        if methods[i]:  # Allows us to apply to only methods of interest
            method = method_names[i]

            for cond_name in cond_names:  # Conditions (median, tibial)
                evoked_list = []

                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'

                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                # Changing to pick channel when each is loaded, not after the evoked list is formed
                xls = pd.ExcelFile('/data/p_02569/Components.xls')
                df = pd.read_excel(xls, 'Dataset 1')
                df.set_index('Subject', inplace=True)
                # channel = ['Cor1', 'Cor2']

                for subject in subjects:  # All subjects
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    if method == 'Prep':
                        input_path = "/data/pt_02569/tmp_data/prepared_py_cca/" + subject_id
                        epochs = mne.read_epochs(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                                                 , preload=True)
                        channel = df.loc[subject_id, f"Prep_{cond_name}"]
                        inv = df.loc[subject_id, f"Prep_{cond_name}_inv"]
                        if inv == 'inv' or inv == '!inv':
                            epochs.apply_function(invert, picks=channel)
                        evoked = epochs.average(picks=[channel])
                        data = evoked.data
                        evoked_list.append(data)

                    elif method == 'PCA':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py_cca/" + subject_id
                        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                        epochs = mne.read_epochs(input_path + fname, preload=True)
                        channel = df.loc[subject_id, f"PCA_{cond_name}"]
                        inv = df.loc[subject_id, f"PCA_{cond_name}_inv"]
                        if inv == 'inv' or inv == '!inv':
                            epochs.apply_function(invert, picks=channel)
                        evoked = epochs.average(picks=[channel])
                        data = evoked.data
                        evoked_list.append(data)

                average = np.mean(evoked_list, axis=0)
                plt.plot(epochs.times, average[0, :],
                         label='Evoked Grand Average')
                plt.ylabel('Amplitude [a.u.]')

                plt.xlabel('Time [s]')
                plt.xlim([-0.1, 0.3])
                if cond_name == 'tibial':
                    plt.axvline(x=22 / 1000, color='r', linewidth=0.5, label='22ms')
                elif cond_name == 'median':
                    plt.axvline(x=13 / 1000, color='r', linewidth=0.5, label='13ms')
                plt.title(f"Method: {method}, Condition: {trigger_name}, CCA: True")
                fname = f"{method}_{trigger_name}_cca_separated.png"
                plt.legend(loc='upper right')
                plt.savefig(image_path+fname)
                plt.clf()

    # Now deal with SSP plots - Just doing 5 to 10 for now
    for n in np.arange(5, 7):  # Methods Applied
        for cond_name in cond_names:  # Conditions (median, tibial)
            evoked_list = []

            if cond_name == 'tibial':
                trigger_name = 'Tibial - Stimulation'

            elif cond_name == 'median':
                trigger_name = 'Median - Stimulation'

            xls = pd.ExcelFile('/data/p_02569/Components.xls')
            df = pd.read_excel(xls, 'Dataset 1')
            df.set_index('Subject', inplace=True)
            # channel = ['Cor1', 'Cor2']

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                input_path = f"/data/pt_02569/tmp_data/ssp_py_cca/{subject_id}/{n} projections/"
                epochs = mne.read_epochs(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                channel = df.loc[subject_id, f"SSP{n}_{cond_name}"]
                inv = df.loc[subject_id, f"SSP{n}_{cond_name}_inv"]
                if inv == 'inv' or inv == '!inv':
                    epochs.apply_function(invert, picks=channel)
                evoked = epochs.average(picks=[channel])
                data = evoked.data
                evoked_list.append(data)

            average = np.mean(evoked_list, axis=0)
            plt.plot(epochs.times, average[0, :],
                     label='Evoked Grand Average')
            plt.ylabel('Amplitude [A.U.]')

            plt.xlabel('Time [s]')
            plt.xlim([-0.1, 0.3])
            if cond_name == 'tibial':
                plt.axvline(x=22 / 1000, color='r', linewidth=0.5, label='22ms')
            elif cond_name == 'median':
                plt.axvline(x=13 / 1000, color='r', linewidth=0.5, label='13ms')
            plt.title(f"Method: SSP {n} proj., Condition: {trigger_name}, CCA: True")
            fname = f"SSP_{n}_{trigger_name}_cca_separated.png"
            plt.legend(loc='upper right')
            plt.savefig(image_path + fname)
            plt.clf()




