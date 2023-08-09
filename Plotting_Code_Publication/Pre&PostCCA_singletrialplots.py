######################## Plot image for cca_epochs and uncleaned data ############################

import mne
import numpy as np

import invert
from transform import transform
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import pandas as pd
import random

if __name__ == '__main__':
    reduced_trials = True
    if reduced_trials:
        no = 1000
        trial_indices = random.sample(range(1999), no)  # Need to be all unique to avoid selecting the same trials
        trial_indices.sort()  # Want in chronological order

    SSP_proj = 6
    # subjects = [1, 2, 6, 7, 14, 15, 30, 32]
    subjects = [6]
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    # Contains information on which CCA component to pick
    xls = pd.ExcelFile('/data/p_02569/Components.xls')
    df = pd.read_excel(xls, 'Dataset 1')
    df.set_index('Subject', inplace=True)

    for method in ['Uncleaned_PreCCA', 'PCA-OBS_PreCCA', 'SSP_PreCCA', 'Uncleaned', 'PCA', 'SSP']:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path_st = f'/data/p_02569/1ComponentSinglePlots_Dataset1/WithUncleaned/{subject_id}/'
            os.makedirs(figure_path_st, exist_ok=True)

            for cond_name in cond_names:
                if cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channel = 'SC6'
                    pot = 'N13'
                    pot_lat = 0.013
                elif cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channel = 'L1'
                    pot = 'N22'
                    pot_lat = 0.022

                if method == 'Uncleaned_PreCCA':
                    input_path = '/data/pt_02569/tmp_data/prepared_py/' + subject_id + \
                                 '/esg/prepro/epochs_' + cond_name + '.fif'
                    epochs = mne.read_epochs(input_path, preload=True)

                elif method == 'PCA-OBS_PreCCA':
                    input_path = '/data/pt_02569/tmp_data/ecg_rm_py/' + subject_id + \
                                        '/esg/prepro/epochs_' + cond_name + '.fif'
                    epochs = mne.read_epochs(input_path, preload=True)

                elif method == 'SSP_PreCCA':
                    input_path = "/data/p_02569/SSP/" + subject_id + f"/6 projections/epochs_" + cond_name + ".fif"
                    epochs = mne.read_epochs(input_path, preload=True)

                elif method == 'Uncleaned':
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

                elif method == 'SSP':
                    input_path = f"/data/p_02569/SSP_cca/{subject_id}/{SSP_proj} projections/"
                    epochs = mne.read_epochs(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    channel = df.loc[subject_id, f"SSP{SSP_proj}_{cond_name}"]
                    inv = df.loc[subject_id, f"SSP{SSP_proj}_{cond_name}_inv"]
                    if inv == 'inv' or inv == '!inv':
                        epochs.apply_function(invert, picks=channel)

                fig, ax = plt.subplots(figsize=(5.2, 7))
                cropped = epochs.copy().crop(tmin=-0.025, tmax=0.065)

                # Z-score transform each trial
                epochs.apply_function(transform, picks=channel)

                if reduced_trials:
                    cropped = cropped[trial_indices]  # Select epochs of interest

                # cmap = mpl.colors.ListedColormap(["mediumblue", "deepskyblue", "lemonchiffon", "gold"])
                cmap = 'plasma'
                vmin = -0.8
                vmax = 0.8

                if method == 'Uncleaned_PreCCA' or method == 'PCA-OBS_PreCCA' or method == 'SSP_PreCCA':
                    cropped.plot_image(picks=channel, combine=None, cmap=cmap, evoked=False, show=False,
                                       title=f"{method.replace('_PreCCA', '')}, Channel {channel}",
                                       colorbar=False, group_by=None, fig=fig,
                                       vmin=vmin, vmax=vmax)
                elif method == 'SSP':
                    cropped.plot_image(picks=channel, combine=None, cmap=cmap, evoked=False, show=False,
                                       title=f'{method} + CCA, Component {str(channel)[-1]}',
                                       colorbar=False, group_by=None, fig=fig,
                                       vmin=vmin, vmax=vmax, scalings=dict(eeg=1))
                elif method == 'PCA':
                    cropped.plot_image(picks=channel, combine=None, cmap=cmap, evoked=False, show=False,
                                       title=f'{method}-OBS + CCA, Component {str(channel)[-1]}',
                                       colorbar=False, group_by=None, fig=fig,
                                       vmin=vmin, vmax=vmax, scalings=dict(eeg=1))
                else:
                    cropped.plot_image(picks=channel, combine=None, cmap=cmap, evoked=False, show=False,
                                       title=f'{method} + CCA, Component {str(channel)[-1]}',
                                       colorbar=False, group_by=None, fig=fig,
                                       vmin=vmin, vmax=vmax, scalings=dict(eeg=1))

                # ax.annotate(pot, xy=(pot_lat, 0),
                #             arrowprops=dict(color='red', facecolor='red'))

                # plt.xticks([pot_lat], [pot], color='red')
                plt.plot(pot_lat, -8, '^', clip_on=False, color='red')
                # ax.arrow(x=pot_lat, y=0, dx=100, dy=50)
                # ax.annotate(pot,
                #             xy=(pot_lat, 0),
                #             xytext=(pot_lat, -0.1),
                #             ha='center',
                #             va='center',
                #             arrowprops={'arrowstyle': '->'})

                plt.tight_layout()
                fig.subplots_adjust(right=0.85)
                ax5 = fig.add_axes([0.9, 0.1, 0.01, 0.8])
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                mpl.colorbar.ColorbarBase(ax5, cmap=cmap, norm=norm)

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