######################## Plot image for cca_epochs ############################
# Just looking at

import mne
import invert
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from scipy.io import loadmat
import numpy as np
import pandas as pd
import random

if __name__ == '__main__':
    reduced_trials = True
    if reduced_trials:
        no = 500
        trial_indices = random.sample(range(1999), no)  # Need to be all unique to avoid selecting the same trials
        trial_indices.sort()  # Want in chronological order

    SSP_proj = 6
    # Testing with just subject 1 at the moment
    # subjects = np.arange(1, 37)  # (1, 37) # 1 through 36 to access subject data
    subjects = [1, 2, 6, 7, 14, 15, 30, 32]
    # subjects = [6]
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    # Contains information on which CCA component to pick
    xls = pd.ExcelFile('/data/p_02569/Components.xls')
    df = pd.read_excel(xls, 'Dataset 1')
    df.set_index('Subject', inplace=True)

    for method in ['Uncleaned', 'PCA', 'Post-ICA', 'SSP']:
        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                elif cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                if method == 'Uncleaned':
                    input_path = "/data/pt_02569/tmp_data/prepared_py_cca/" + subject_id + "/esg/prepro/"
                    epochs = mne.read_epochs(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                                             , preload=True)
                    channel = df.loc[subject_id, f"Prep_{cond_name}"]
                    inv = df.loc[subject_id, f"Prep_{cond_name}_inv"]
                    if inv == 'inv' or inv == '!inv':
                        epochs.apply_function(invert, picks=channel)

                elif method == 'PCA':
                    input_path = "/data/pt_02569/tmp_data/ecg_rm_py_cca/" + subject_id + "/esg/prepro/"
                    fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    channel = df.loc[subject_id, f"PCA_{cond_name}"]
                    inv = df.loc[subject_id, f"PCA_{cond_name}_inv"]
                    if inv == 'inv' or inv == '!inv':
                        epochs.apply_function(invert, picks=channel)

                elif method == 'Post-ICA':
                    input_path = "/data/pt_02569/tmp_data/ica_py_cca/" + subject_id + "/esg/prepro/"
                    fname = f"clean_ica_auto_{cond_name}.fif"
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    channel = df.loc[subject_id, f"Post-ICA_{cond_name}"]
                    inv = df.loc[subject_id, f"Post-ICA_{cond_name}_inv"]
                    if inv == 'inv' or inv == '!inv':
                        epochs.apply_function(invert, picks=channel)

                elif method == 'SSP':
                    input_path = f"/data/p_02569/SSP_cca/{subject_id}/{SSP_proj} projections/"
                    epochs = mne.read_epochs(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    channel = df.loc[subject_id, f"SSP{SSP_proj}_{cond_name}"]
                    inv = df.loc[subject_id, f"SSP{SSP_proj}_{cond_name}_inv"]
                    if inv == 'inv' or inv == '!inv':
                        epochs.apply_function(invert, picks=channel)

                # cca_epochs and cca_epochs_d both already baseline corrected before this point
                figure_path_st = f'/data/p_02569/1ComponentSinglePlots_Dataset1/{subject_id}/'
                os.makedirs(figure_path_st, exist_ok=True)

                fig, ax = plt.subplots(figsize=(5.2, 7))
                cropped = epochs.copy().crop(tmin=-0.025, tmax=0.065)
                if reduced_trials:
                    cropped = cropped[trial_indices]  # Select epochs of interest

                cmap = mpl.colors.ListedColormap(["mediumblue", "deepskyblue", "lemonchiffon", "gold"])

                if method == 'SSP':
                    cropped.plot_image(picks=channel, combine=None, cmap=cmap, evoked=False, show=False,
                                       title=f'{method}{SSP_proj}, Subject {subject}, Component {str(channel)[-1]}',
                                       colorbar=False, group_by=None, fig=fig,
                                       vmin=-0.4, vmax=0.4, units=dict(eeg='V'), scalings=dict(eeg=1))
                else:
                    cropped.plot_image(picks=channel, combine=None, cmap=cmap, evoked=False, show=False,
                                       title=f'{method}, Subject {subject}, Component {str(channel)[-1]}',
                                       colorbar=False, group_by=None, fig=fig,
                                       vmin=-0.4, vmax=0.4, units=dict(eeg='V'), scalings=dict(eeg=1))

                plt.tight_layout()
                fig.subplots_adjust(right=0.85)
                ax5 = fig.add_axes([0.9, 0.1, 0.01, 0.8])
                norm = mpl.colors.Normalize(vmin=-0.4, vmax=0.4)
                mpl.colorbar.ColorbarBase(ax5, cmap=cmap, norm=norm, spacing='proportional')
                # has to be as a list - starts with x, y coordinates for start and then width and height in % of figure width
                if reduced_trials:
                    if method == 'SSP':
                        # plt.suptitle(f'{data_string}_{n}_{cond_name_mixed}')
                        plt.savefig(figure_path_st + f'{method}_{SSP_proj}_{cond_name}_reducedtrials.png')

                    else:
                        # plt.suptitle(f'{data_string}_{cond_name_mixed}')
                        plt.savefig(figure_path_st + f'{method}_{cond_name}_reducedtrials.png')
                else:
                    if method == 'SSP':
                        # plt.suptitle(f'{data_string}_{n}_{cond_name_mixed}')
                        plt.savefig(figure_path_st + f'{method}_{SSP_proj}_{cond_name}.png')

                    else:
                        # plt.suptitle(f'{data_string}_{cond_name_mixed}')
                        plt.savefig(figure_path_st + f'{method}_{cond_name}.png')
                plt.close(fig)
