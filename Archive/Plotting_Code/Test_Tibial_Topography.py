# Want a 3x4 plot matrix (Timecourse, TFR, Topography) for each of (Uncleaned, PCA_OBS, ICA, SSP6)

import numpy as np
import mne
import matplotlib.pyplot as plt
from Plotting_Code_Publication.IsopotentialFunctions_Axis import mrmr_esg_isopotentialplot
from get_esg_channels import get_esg_channels
from scipy.io import loadmat
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()
    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']
    #  First get the evoked list
    # trigger_names = ['Tibial - Stimulation', 'Median - Stimulation']
    trigger_names = ['Tibial - Stimulation']
    save_path = '/data/p_02569/Images/SEP_Test/'
    os.makedirs(save_path, exist_ok=True)

    # methods = ['Uncleaned', 'PCA']
    methods = ['PCA']

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    # epoch
    iv_epoch = [-200 / 1000, 700 / 1000]
    # let's say we epoch the data from -200 to 700 ms around the trigger of interest
    iv_baseline = [-100 / 1000, -10 / 1000]

    subjects = np.arange(1, 37)
    # subjects = [1]

    # trials = [True, False, True, False]
    # time = [False, True, True, False]

    trials = [True]
    time = [False]

    for reduced_trials, shorter_timescale in zip(trials, time):
        for trigger_name in trigger_names:
            count = 0  # To cycle through the subplots
            fig, axes = plt.subplots(2, 1, figsize=[18, 24])
            axes_unflat = axes
            axes = axes.flatten()

            for method in methods:
                evoked_list = []
                data_list = []

                for subj in subjects:
                    subject_id = f'sub-{str(subj).zfill(3)}'

                    potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
                    fname_pot = 'potential_latency.mat'
                    matdata = loadmat(potential_path + fname_pot)

                    if trigger_name == 'Median - Stimulation':
                        time_point = 13 / 1000
                        cond_name = 'median'
                        channel = ['SC6']
                        # sep_latency = matdata['med_potlatency']
                        # time_point = sep_latency[0][0] / 1000
                    else:
                        cond_name = 'tibial'
                        time_point = 22 / 1000
                        channel = ['L1']
                        # sep_latency = matdata['tib_potlatency']
                        # time_point = sep_latency[0][0] / 1000

                    if method == 'Uncleaned':
                        data_path = '/data/pt_02569/tmp_data/prepared_py/' + subject_id + \
                                    '/epochs_' + cond_name + '.fif'

                    elif method == 'PCA':
                        data_path = '/data/pt_02569/tmp_data/ecg_rm_py/' + subject_id + \
                                        '/epochs_' + cond_name + '.fif'

                    # For all others, read in the epochs we have constructed
                    epochs = mne.read_epochs(data_path, preload=True).reorder_channels(esg_chans)
                    # if 'TH6' in epochs.ch_names:
                    #     epochs = epochs.copy().drop_channels('TH6')
                    # if 'Fz-TH6' in epochs.ch_names:
                    #     epochs = epochs.copy().drop_channels('Fz-TH6')
                    # Want each channel averaged across all epochs at a given time point
                    if reduced_trials:
                        epochs = epochs[0::4]
                    if subj == 34:
                        evoked = epochs.average().crop(tmin=time_point, tmax=time_point + (2 / 1000))
                        data = evoked.data.mean(axis=1)
                        ch_idx = evoked.ch_names.index("S34")
                        data = np.insert(data, ch_idx, np.nan)
                        data_list.append(data)
                    else:
                        evoked = epochs.average().crop(tmin=time_point, tmax=time_point + (2 / 1000))
                        data = evoked.data.mean(axis=1)
                        data_list.append(data)

                #     evoked_list.append(epochs.average())  # Check if averages make sense
                #
                # evoked = mne.grand_average(evoked_list)

                #########################################################################################
                # Plot Time Courses for relevant patch
                #########################################################################################
                # if trigger_name == 'Tibial - Stimulation':
                #     relevant_channels = evoked.copy().pick_channels(lumbar_chans)
                # relevant_channels.plot(exclude=[])

                ##########################################################################################
                # Plot the topography
                ##########################################################################################
                # time_idx = []
                # tmp = np.argwhere(epochs.times >= time_point)
                # # sometimes when data is down sampled  find(epo.times == time_points(ii)) doesn't work
                # time_idx.append(tmp[0])
                # chanvalues = evoked.data[:, time_idx]
                # chan_labels = evoked.ch_names

                # chanvalues = evoked.crop(tmin=time_point, tmax=time_point+(2/1000)).data
                # chanvalues = chanvalues.mean(axis=1)  # Get average in window of interest
                arrays = [np.array(x) for x in data_list]
                chanvalues = np.array([np.nanmean(k) for k in zip(*arrays)])

                # chan_labels = evoked.ch_names
                chan_labels = esg_chans
                colorbar_axes = [-0.5, 0.5]
                subjects_4grid = np.arange(1, 37)
                # then the function takes the average over the channel positions of all those subjects
                colorbar = True
                mrmr_esg_isopotentialplot(subjects_4grid, chanvalues, colorbar_axes, chan_labels, colorbar,
                                          time_point, axes[count])
                axes[count].set_xticklabels([])
                axes[count].set_yticklabels([])
                axes[count].set_xticks([])
                axes[count].set_yticks([])
                if count == 2:
                    axes[count].set_title('Spatial Topographies')
                count += 1

            # Final Formatting
            plt.suptitle('Somatosensory Evoked Potentials', x=0.55)
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, top=0.95)
            if reduced_trials and shorter_timescale:
                fname = f"Test_Combined_{cond_name}_reducedtrials_shorter"
            elif reduced_trials and not shorter_timescale:
                fname = f"Test_Combined_{cond_name}_reducedtrials"
            elif shorter_timescale and not reduced_trials:
                fname = f"Test_Combined_{cond_name}_shorter"
            else:
                fname = f"Test_Combined_{cond_name}"

            plt.savefig(save_path+fname+'.png')
            plt.savefig(save_path+fname+'.pdf', bbox_inches='tight', format="pdf")
            # plt.show()


