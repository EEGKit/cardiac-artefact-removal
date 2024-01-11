# Script to create plots of the single subject evoked responses of the heartbeat
# for each stimulation and cleaning method

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_axes_aligner import align
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    pal = sns.color_palette(n_colors=4)
    # subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    subjects = np.arange(1, 37)  # [1, 2, 6, 20, 31, 34]  # , 6, 20, 31
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

    image_path = "/data/p_02569/SingleSubjecICAtHeartTest_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    for cond_name in cond_names:  # Conditions (median, tibial)
        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channel = 'L1'

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channel = 'SC6'

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked_ica = epochs.average()
            evoked_ica.reorder_channels(esg_chans)

            relevant_channel_ica = evoked_ica.pick_channels([channel])

            # Want 1 row, 3 column subplot
            # Want left y-axis to relate to cleaned heart artefact
            # Want right y-axis to relate to uncleaned heart artefact
            fig, ax = plt.subplots(1, 1)

            # ICA
            ax.plot(relevant_channel_ica.times, relevant_channel_ica.data[0, :] * 10 ** 6, label='ICA',
                     color=pal[2])
            ax.set_xlabel('Time (s)')
            if cond_name == 'median':
                ax.set_title('ICA')
            ax.set_ylabel('Amplitude (uV)')
            ax.set_xlim([-200/1000, 400/1000])

            fname = f"{subject_id}_ICAHeartTimeCourse_{channel}.png"
            plt.tight_layout()
            plt.savefig(image_path+fname)

