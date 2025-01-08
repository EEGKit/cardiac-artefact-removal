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
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

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

    for subject in subjects:
        for cond_name in cond_names:
            fig = plt.figure(figsize=(16, 9))
            gs = fig.add_gridspec(2, 4, width_ratios=[5, 5, 5, 0.25])
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[1, 0])
            ax5 = fig.add_subplot(gs[1, 1])
            ax6 = fig.add_subplot(gs[1, 2])
            cbar_ax = fig.add_subplot(gs[0:2, 3])

            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path_st = f'/data/p_02569/Images/1ComponentSinglePlots_Dataset1/WithUncleaned_Separated/{subject_id}/'
            os.makedirs(figure_path_st, exist_ok=True)

            for method in ['Uncleaned_PreCCA', 'PCA-OBS_PreCCA', 'SSP_PreCCA', 'Uncleaned', 'PCA', 'SSP']:
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
                                 '/epochs_' + cond_name + '.fif'
                    epochs = mne.read_epochs(input_path, preload=True)
                    tit = f"{method.replace('_PreCCA', '')}"  # Channel {channel}
                    ax = ax1

                elif method == 'PCA-OBS_PreCCA':
                    input_path = '/data/pt_02569/tmp_data/ecg_rm_py/' + subject_id + \
                                        '/epochs_' + cond_name + '.fif'
                    epochs = mne.read_epochs(input_path, preload=True)
                    tit = f"{method.replace('_PreCCA', '')}"
                    ax = ax2

                elif method == 'SSP_PreCCA':
                    input_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id + f"/6 projections/epochs_" + cond_name + ".fif"
                    epochs = mne.read_epochs(input_path, preload=True)
                    tit = f"{method.replace('_PreCCA', '')}"
                    ax = ax3

                elif method == 'Uncleaned':
                    input_path = "/data/pt_02569/tmp_data/prepared_py_cca/" + subject_id
                    epochs = mne.read_epochs(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                                             , preload=True)
                    ax = ax4
                    channel = df.loc[subject_id, f"Prep_{cond_name}"]
                    tit = f'{method} + CCA'
                    inv = df.loc[subject_id, f"Prep_{cond_name}_inv"]
                    if inv == 'inv' or inv == '!inv':
                        epochs.apply_function(invert, picks=channel)

                elif method == 'PCA':
                    input_path = "/data/pt_02569/tmp_data/ecg_rm_py_cca/" + subject_id
                    fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    ax = ax5
                    channel = df.loc[subject_id, f"PCA_{cond_name}"]
                    tit = f'{method}-OBS + CCA'
                    inv = df.loc[subject_id, f"PCA_{cond_name}_inv"]
                    if inv == 'inv' or inv == '!inv':
                        epochs.apply_function(invert, picks=channel)

                elif method == 'SSP':
                    input_path = f"/data/pt_02569/tmp_data/ssp_py_cca/{subject_id}/{SSP_proj} projections/"
                    epochs = mne.read_epochs(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    ax = ax6
                    channel = df.loc[subject_id, f"SSP{SSP_proj}_{cond_name}"]
                    tit = f'{method} + CCA'
                    # tit = f'{method} + CCA, Component {str(channel)[-1]}'
                    inv = df.loc[subject_id, f"SSP{SSP_proj}_{cond_name}_inv"]
                    if inv == 'inv' or inv == '!inv':
                        epochs.apply_function(invert, picks=channel)

                # fig, ax = plt.subplots(figsize=(5.2, 7))
                cropped = epochs.copy().crop(tmin=-0.025, tmax=0.065)

                # Z-score transform each trial
                cropped.apply_function(transform, picks=channel)

                if reduced_trials:
                    cropped = cropped[trial_indices]  # Select epochs of interest

                cmap = 'plasma'
                vmin = -1
                vmax = 1

                cropped.plot_image(picks=channel, combine=None, cmap=cmap, evoked=False, show=False,
                                   colorbar=False, group_by=None,
                                   vmin=vmin, vmax=vmax, axes=ax, title=tit, scalings=dict(eeg=1))

                # ax.annotate(pot, xy=(pot_lat, 0),
                #             arrowprops=dict(color='red', facecolor='red'))

                # plt.xticks([pot_lat], [pot], color='red')
                ax.plot(pot_lat, -8, '^', clip_on=False, color='red')
                # ax.arrow(x=pot_lat, y=0, dx=100, dy=50)
                # ax.annotate(pot,
                #             xy=(pot_lat, 0),
                #             xytext=(pot_lat, -0.1),
                #             ha='center',
                #             va='center',
                #             arrowprops={'arrowstyle': '->'})

            # plt.tight_layout()
            # fig.subplots_adjust(right=0.85)
            # ax5 = fig.add_axes([0.9, 0.1, 0.01, 0.8])
            # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            # mpl.colorbar.ColorbarBase(ax5, cmap=cmap, norm=norm)
            # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            # mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
            cb = fig.colorbar(ax6.images[-1], cax=cbar_ax)

            plt.tight_layout()
            if reduced_trials:
                plt.savefig(figure_path_st + f'{cond_name}_reducedtrials.png')
                plt.savefig(figure_path_st+f'{cond_name}_reducedtrials.pdf', bbox_inches='tight', format="pdf")
            else:
                plt.savefig(figure_path_st + f'{cond_name}.png')
                plt.savefig(figure_path_st+f'{cond_name}.pdf', bbox_inches='tight', format="pdf")
            plt.close(fig)