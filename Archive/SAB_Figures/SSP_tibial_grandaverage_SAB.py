# Script to create plots of the grand averages evoked responses across participants for each stimulation

import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw, evoked_from_fourth_raw
import matplotlib.pyplot as plt
import seaborn
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    longer_time = True
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    # subjects = [1, 2]
    cond_names = ['tibial', 'median']
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

    image_path = "/data/p_02569/Images/GrandAveragePlots_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    SSP = True

    # Now deal with SSP plots - Just doing 5 to 10 for now
    if SSP:
        for n in np.arange(6, 7):  # Methods Applied
            for cond_name in cond_names:  # Conditions (median, tibial)
                evoked_list = []

                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channel = 'L1'

                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channel = 'SC6'

                for subject in subjects:  # All subjects
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    input_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/{n} projections/"
                    raw = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                    evoked.reorder_channels(esg_chans)
                    evoked = evoked.pick_channels([channel], ordered=True)
                    evoked_list.append(evoked)

                relevant_channel = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
                # relevant_channel = averaged.pick_channels(channel)
                fig, ax = plt.subplots(1, 1)
                seaborn.set(color_codes=True)
                plt.plot(relevant_channel.times, np.mean(relevant_channel.data[:, :], axis=0) * 10 ** 6, color='r')
                plt.ylabel('Amplitude (\u03BCV)')
                plt.xlabel('Time (s)')
                plt.xlim([-0.1, 0.3])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fname = f"ASAB_SSP_{n}_{trigger_name}.png"
                plt.savefig(image_path + fname)
                plt.savefig(image_path + fname+'.pdf', bbox_inches='tight', format="pdf")
                plt.clf()

    plt.show()
