# Script to create plots of single subject evoked responses across participants for
# each stimulation and cleaning regime

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from mpl_axes_aligner import align

def align_yaxis_np(axes):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array(axes)
    extrema = np.array([ax.get_ylim() for ax in axes])

    # reset for divide by zero issues
    for i in range(len(extrema)):
        if np.isclose(extrema[i, 0], 0.0):
            extrema[i, 0] = -1
        if np.isclose(extrema[i, 1], 0.0):
            extrema[i, 1] = 1

    # upper and lower limits
    lowers = extrema[:, 0]
    uppers = extrema[:, 1]

    # if all pos or all neg, don't scale
    all_positive = False
    all_negative = False
    if lowers.min() > 0.0:
        all_positive = True

    if uppers.max() < 0.0:
        all_negative = True

    if all_negative or all_positive:
        # don't scale
        return

    # pick "most centered" axis
    res = abs(uppers+lowers)
    min_index = np.argmin(res)

    # scale positive or negative part
    multiplier1 = abs(uppers[min_index]/lowers[min_index])
    multiplier2 = abs(lowers[min_index]/uppers[min_index])

    for i in range(len(extrema)):
        # scale positive or negative part based on which induces valid
        if i != min_index:
            lower_change = extrema[i, 1] * -1*multiplier2
            upper_change = extrema[i, 0] * -1*multiplier1
            if upper_change < extrema[i, 1]:
                extrema[i, 0] = lower_change
            else:
                extrema[i, 1] = upper_change

        # bump by 10% for a margin
        extrema[i, 0] *= 1.1
        extrema[i, 1] *= 1.1

    # set axes limits
    [axes[i].set_ylim(*extrema[i]) for i in range(len(extrema))]


if __name__ == '__main__':
    pal = sns.color_palette(n_colors=4)
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    # subjects = [1, 6, 20, 31, 34]
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/Images/SingleSubjectYY_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    trials = [True, False, True, False]
    time = [False, True, True, False]

    for reduced_trials, shorter_timescale in zip(trials, time):
        for cond_name in cond_names:  # Conditions (median, tibial)

            if cond_name == 'tibial':
                trigger_name = 'Tibial - Stimulation'
                channel = 'L1'

            elif cond_name == 'median':
                trigger_name = 'Median - Stimulation'
                channel = 'SC6'

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                ################################################################################
                # Uncleaned
                ###############################################################################
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
                fname = f"epochs_{cond_name}.fif"
                epochs = mne.read_epochs(input_path+fname, preload=True)
                if reduced_trials:
                    epochs = epochs[0::4]
                evoked_uncleaned = epochs.average()
                evoked_uncleaned.reorder_channels(esg_chans)

                ##############################################################################
                # PCA_OBS
                ##############################################################################
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
                fname = f"epochs_{cond_name}.fif"
                epochs = mne.read_epochs(input_path + fname, preload=True)
                if reduced_trials:
                    epochs = epochs[0::4]
                evoked_pca = epochs.average()
                evoked_pca.reorder_channels(esg_chans)

                ##############################################################################
                # ICA
                ##############################################################################
                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + '/'
                fname = f"epochs_{cond_name}.fif"
                epochs = mne.read_epochs(input_path + fname, preload=True)
                if reduced_trials:
                    epochs = epochs[0::4]
                evoked_ica = epochs.average()
                evoked_ica.reorder_channels(esg_chans)

                #############################################################################
                # SSP 6
                #############################################################################
                input_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/6 projections/"
                fname = f"epochs_{cond_name}.fif"
                epochs = mne.read_epochs(input_path + fname, preload=True)
                if reduced_trials:
                    epochs = epochs[0::4]
                evoked_ssp6 = epochs.average()
                evoked_ssp6.reorder_channels(esg_chans)

                #################################################################################
                # Get relevant channels
                #################################################################################
                relevant_channel_prep = evoked_uncleaned.pick_channels([channel])
                relevant_channel_pca = evoked_pca.pick_channels([channel])
                relevant_channel_ica = evoked_ica.pick_channels([channel])
                relevant_channel_ssp6 = evoked_ssp6.pick_channels([channel])

                # Want 1 row, 3 column subplot
                # Want left y-axis to relate to cleaned heart artefact
                # Want right y-axis to relate to uncleaned heart artefact
                fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=[18, 6])
                ax10 = ax1.twinx()
                ax20 = ax2.twinx()
                ax30 = ax3.twinx()
                # ax1.get_shared_y_axes().join(ax1, ax2, ax3)  # Tie primary axes
                # ax2.get_shared_y_axes().join(ax2, ax3)
                ax10.get_shared_y_axes().join(ax10, ax20, ax30)  # Tie secondary axes

                # PCA
                ax1.plot(relevant_channel_pca.times, relevant_channel_pca.data[0, :]*10**6, label='PCA-OBS',
                         color=pal[1])
                ax1.set_ylabel('Cleaned SEP Amplitude (\u03BCV)')
                ax1.set_xlabel('Time (s)')
                if cond_name == 'median':
                    ax1.set_title('PCA-OBS')
                ax1.spines['left'].set_color(pal[1])
                ax1.tick_params(axis='y', colors=pal[1])
                ax10.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :]*10**6, label='Uncleaned',
                          linewidth=0.5, linestyle='dashed', color='blue')
                ax10.set_yticklabels([])

                # ICA
                ax2.plot(relevant_channel_ica.times, relevant_channel_ica.data[0, :] * 10 ** 6, label='ICA',
                         color=pal[2])
                ax2.set_xlabel('Time (s)')
                if cond_name == 'median':
                    ax2.set_title('ICA')
                # ax2.set_yticklabels([])
                ax2.spines['left'].set_color(pal[2])
                ax2.tick_params(axis='y', colors=pal[2])
                ax20.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :] * 10 ** 6, label='Uncleaned',
                          linewidth=0.5, linestyle='dashed', color='blue')
                ax20.set_yticklabels([])

                # SSP6
                ax3.plot(relevant_channel_ssp6.times, relevant_channel_ssp6.data[0, :] * 10 ** 6, label='SSP',
                         color=pal[3])
                ax3.set_xlabel('Time (s)')
                if cond_name == 'median':
                    ax3.set_title('SSP')
                # ax3.set_yticklabels([])
                ax30.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :] * 10 ** 6, label='Uncleaned',
                          linewidth=0.5, linestyle='dashed', color='blue')
                ax30.set_ylabel('Uncleaned SEP Amplitude (\u03BCV)')
                ax3.spines['left'].set_color(pal[3])
                ax3.tick_params(axis='y', colors=pal[3])
                # ax30.spines['right'].set_color('blue')
                ax30.yaxis.label.set_color('blue')
                ax30.tick_params(axis='y', colors='blue')

                # Add vertical line at expected latency
                if cond_name == 'tibial':
                    ax1.axvline(x=22 / 1000, color='g', linewidth=0.7, label='22ms')
                    ax2.axvline(x=22 / 1000, color='g', linewidth=0.7, label='22ms')
                    ax3.axvline(x=22 / 1000, color='g', linewidth=0.7, label='22ms')

                elif cond_name == 'median':
                    ax1.axvline(x=13 / 1000, color='g', linewidth=0.7, label='13ms')
                    ax2.axvline(x=13 / 1000, color='g', linewidth=0.7, label='13ms')
                    ax3.axvline(x=13 / 1000, color='g', linewidth=0.7, label='13ms')

                if shorter_timescale:
                    ax1.set_xlim([-25 / 1000, 65 / 1000])
                    ax2.set_xlim([-25 / 1000, 65 / 1000])
                    ax3.set_xlim([-25 / 1000, 65 / 1000])
                else:
                    ax1.set_xlim([-100/1000, 300/1000])
                    ax2.set_xlim([-100/1000, 300/1000])
                    ax3.set_xlim([-100/1000, 300/1000])

                if reduced_trials and shorter_timescale:
                    fname = f"{subject_id}_SEPTimeCourse_{channel}_reducedtrials_shorter.png"
                elif reduced_trials and not shorter_timescale:
                    fname = f"{subject_id}_SEPTimeCourse_{channel}_reducedtrials.png"
                elif shorter_timescale and not reduced_trials:
                    fname = f"{subject_id}_SEPTimeCourse_{channel}_shorter.png"
                else:
                    fname = f"{subject_id}_SEPTimeCourse_{channel}.png"

                # # Align y-axes
                align_yaxis_np([ax1, ax10, ax2, ax20, ax3, ax30])

                # if cond_name == 'median':
                #     plt.suptitle(f"SEP Time Courses\n"
                #                  f"Cervical Spinal Cord\n"
                #                  f"Subject {subject}")
                # else:
                #     plt.suptitle(f"SEP Time Courses\n"
                #                  f"Lumbar Spinal Cord\n"
                #                  f"Subject {subject}")
                plt.tight_layout()
                # plt.show()
                plt.savefig(image_path+fname)
                plt.savefig(image_path+fname+'.pdf', bbox_inches='tight', format="pdf")


