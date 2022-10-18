# Script to create plots of the grand averages evoked responses of the heartbeat across participants for each stimulation

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_axes_aligner import align


if __name__ == '__main__':
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
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

    image_path = "/data/p_02569/GrandAverageHeartYY_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list_prep = []
        evoked_list_pca = []
        evoked_list_ica = []
        evoked_list_ssp6 = []

        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channel = 'L1'

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channel = 'SC6'

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path+fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked_list_prep.append(evoked)

            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked_list_pca.append(evoked)

            input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked_list_ica.append(evoked)

            input_path = f"/data/p_02569/SSP/{subject_id}/6 projections/"
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked_list_ssp6.append(evoked)

        averaged_prep = mne.grand_average(evoked_list_prep, interpolate_bads=False, drop_bads=False)
        relevant_channel_prep = averaged_prep.pick_channels([channel])

        averaged_pca = mne.grand_average(evoked_list_pca, interpolate_bads=False, drop_bads=False)
        relevant_channel_pca = averaged_pca.pick_channels([channel])

        averaged_ica = mne.grand_average(evoked_list_ica, interpolate_bads=False, drop_bads=False)
        relevant_channel_ica = averaged_ica.pick_channels([channel])

        averaged_ssp6 = mne.grand_average(evoked_list_ssp6, interpolate_bads=False, drop_bads=False)
        relevant_channel_ssp6 = averaged_ssp6.pick_channels([channel])

        # Want 1 row, 3 column subplot
        # Want left y-axis to relate to cleaned heart artefact
        # Want right y-axis to relate to uncleaned heart artefact
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=[18, 6])
        ax10 = ax1.twinx()
        ax20 = ax2.twinx()
        ax30 = ax3.twinx()
        ax1.get_shared_y_axes().join(ax1, ax2, ax3)
        ax10.get_shared_y_axes().join(ax10, ax20, ax30)

        # PCA
        ax1.plot(relevant_channel_pca.times, relevant_channel_pca.data[0, :]*10**6, label='PCA_OBS',
                 color='blue')
        ax1.set_ylabel('Cleaned Artefact Amplitude (\u03BCV)')
        ax1.set_xlabel('Time (s)')
        ax1.set_title('PCA_OBS')
        # ax1.spines['left'].set_color('blue')
        # ax1.tick_params(axis='y', colors='blue')
        ax10.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :]*10**6, label='Uncleaned',
                  linewidth=0.5, linestyle='dashed', color='black')
        ax10.set_yticklabels([])

        # ICA
        ax2.plot(relevant_channel_ica.times, relevant_channel_ica.data[0, :] * 10 ** 6, label='ICA',
                 color='orange')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('ICA')
        ax2.set_yticklabels([])
        # ax2.spines['left'].set_color('orange')
        # ax2.tick_params(axis='y', colors='orange')
        ax20.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :] * 10 ** 6, label='Uncleaned',
                  linewidth=0.5, linestyle='dashed', color='black')
        ax20.set_yticklabels([])

        # SSP6
        ax3.plot(relevant_channel_ssp6.times, relevant_channel_ssp6.data[0, :] * 10 ** 6, label='SSP6',
                 color='magenta')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('SSP6')
        ax3.set_yticklabels([])
        ax30.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :] * 10 ** 6, label='Uncleaned',
                  linewidth=0.5, linestyle='dashed', color='black')
        ax30.set_ylabel('Uncleaned Artefact Amplitude (\u03BCV)')
        # ax3.spines['left'].set_color('magenta')
        # ax3.tick_params(axis='y', colors='magenta')

        ax1.set_xlim([-200/1000, 400/1000])
        ax2.set_xlim([-200/1000, 400/1000])
        ax3.set_xlim([-200/1000, 400/1000])

        fname = f"CardiacTimeCourse__{channel}.png"

        # Align y-axes
        if cond_name == 'median':
            align.yaxes(ax1, 0, ax10, 0, 0.75)
        else:
            align.yaxes(ax1, 0, ax10, 0, 0.25)

        # Collect labels for legend
        # lines, labels = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # lines3, labels3 = ax3.get_legend_handles_labels()
        # lines30, labels30 = ax30.get_legend_handles_labels()
        # plt.legend(lines + lines2 + lines3 + lines30, labels + labels2 + labels3 + labels30, loc='lower left',
        #            bbox_to_anchor=(1, 0))

        if cond_name == 'median':
            plt.suptitle(f"Cardiac Artefact Time Courses\n"
                         f"Cervical Spinal Cord")
        else:
            plt.suptitle(f"Cardiac Artefact Time Courses\n"
                         f"Lumbar Spinal Cord")
        plt.tight_layout()
        plt.savefig(image_path+fname)

