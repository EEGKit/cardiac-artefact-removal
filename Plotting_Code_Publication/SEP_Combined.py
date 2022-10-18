# Want a 3x4 plot matrix (Timecourse, TFR, Topography) for each of (Uncleaned, PCA_OBS, ICA, SSP6)

import numpy as np
import mne
import matplotlib.pyplot as plt
from IsopotentialFunctions_Axis import mrmr_esg_isopotentialplot
from scipy.io import loadmat
import os


if __name__ == '__main__':
    #  First get the evoked list
    trigger_names = ['Median - Stimulation', 'Tibial - Stimulation']
    save_path = '/data/p_02569/SEP_Combined_D1/'
    os.makedirs(save_path, exist_ok=True)

    methods = ['Uncleaned', 'PCA', 'ICA', 'SSP6']

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

    trials = [True, False, True, False]
    time = [False, True, True, False]

    for reduced_trials, shorter_timescale in zip(trials, time):
        for trigger_name in trigger_names:
            count = 0  # To cycle through the subplots
            fig, axes = plt.subplots(4, 3, figsize=[18, 24])
            axes_unflat = axes
            axes = axes.flatten()

            for method in methods:
                evoked_list = []
                evoked_list_pca_extra = []

                for subj in subjects:
                    if trigger_name == 'Median - Stimulation':
                        time_point = 13 / 1000
                        cond_name = 'median'
                        channel = ['SC6']
                    else:
                        cond_name = 'tibial'
                        time_point = 22 / 1000
                        channel = ['L1']

                    subject_id = f'sub-{str(subj).zfill(3)}'

                    if method == 'Uncleaned':
                        data_path = '/data/pt_02569/tmp_data/prepared_py/' + subject_id + \
                                    '/esg/prepro/epochs_' + cond_name + '.fif'

                    elif method == 'PCA':
                        data_path_raw = '/data/pt_02569/tmp_data/ecg_rm_py/' + subject_id + \
                                    '/esg/prepro/data_clean_ecg_spinal_' + cond_name + '_withqrs.fif'

                        data_path_epochs = '/data/pt_02569/tmp_data/ecg_rm_py/' + subject_id + \
                                        '/esg/prepro/epochs_' + cond_name + '.fif'

                    elif method == 'ICA':
                        data_path = '/data/pt_02569/tmp_data/baseline_ica_py/' + subject_id + \
                                    '/esg/prepro/epochs_' + cond_name + '.fif'

                    elif method == 'SSP6':
                        data_path = "/data/p_02569/SSP/" + subject_id + f"/6 projections/epochs_" + cond_name + ".fif"

                    # for PCA - want to interpolate the hump for the TFR plot
                    if method == 'PCA':
                        # Read raw so we can interpolate the hump at 0 - save as one type of PCA for topography
                        # This will need to be filtered and rereferenced
                        raw = mne.io.read_raw_fif(data_path_raw, preload=True)
                        events, event_dict = mne.events_from_annotations(raw)
                        tstart_esg = -0.007
                        tmax_esg = 0.007
                        mne.preprocessing.fix_stim_artifact(raw, events=events, event_id=event_dict[trigger_name],
                                                            tmin=tstart_esg,
                                                            tmax=tmax_esg, mode='linear', stim_channel=None)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                        events, event_ids = mne.events_from_annotations(raw)
                        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                        epochs_interp = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                         baseline=tuple(iv_baseline), preload=True)
                        # if 'TH6' in epochs_interp.ch_names:
                        #     epochs_interp = epochs_interp.copy().drop_channels('TH6')
                        # if 'Fz-TH6' in epochs_interp.ch_names:
                        #     epochs_interp = epochs_interp.copy().drop_channels('Fz-TH6')
                        if reduced_trials:
                            epochs_interp = epochs_interp[0::4]
                        evoked_list_pca_extra.append(epochs_interp.average())

                        # Also read in the epochs already constructed and save in normal pca evoked list
                        epochs = mne.read_epochs(data_path_epochs, preload=True)
                        # if 'TH6' in epochs.ch_names:
                        #     epochs = epochs.copy().drop_channels('TH6')
                        # if 'Fz-TH6' in epochs.ch_names:
                        #     epochs = epochs.copy().drop_channels('Fz-TH6')
                        if reduced_trials:
                            epochs = epochs[0::4]
                        evoked_list.append(epochs.average())

                    else:
                        # For all others, read in the epochs we have constructed
                        epochs = mne.read_epochs(data_path, preload=True)
                        # if 'TH6' in epochs.ch_names:
                        #     epochs = epochs.copy().drop_channels('TH6')
                        # if 'Fz-TH6' in epochs.ch_names:
                        #     epochs = epochs.copy().drop_channels('Fz-TH6')
                        # Want each channel averaged across all epochs at a given time point
                        if reduced_trials:
                            epochs = epochs[0::4]
                        evoked_list.append(epochs.average())


                ##########################################################################################
                # Plot the time course
                ##########################################################################################
                grand_average = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
                relevant_channel = grand_average.pick_channels(channel, ordered=True)
                axes[count].plot(relevant_channel.times, np.mean(relevant_channel.data[:, :], axis=0) * 10 ** 6)
                axes[count].set_ylabel('Amplitude (\u03BCV)')
                if shorter_timescale:
                    axes[count].set_xlim([-0.025, 0.065])
                else:
                    axes[count].set_xlim([-0.1, 0.3])
                if cond_name == 'tibial':
                    axes[count].axvline(x=22 / 1000, color='r', linewidth=0.5, label='22ms')
                elif cond_name == 'median':
                    axes[count].axvline(x=13 / 1000, color='r', linewidth=0.5, label='13ms')
                if count == 0:
                    axes[count].set_title('Time Courses')
                axes[count].set_xlabel('Time (s)')
                count += 1

                ################################################################################################
                # Plot the TFR
                ################################################################################################
                freqs = np.arange(5., 400., 3.)
                fmin, fmax = freqs[[0, -1]]
                power_list = []
                for index in np.arange(0, len(evoked_list)):
                    if method == 'PCA':
                        power = mne.time_frequency.tfr_stockwell(evoked_list_pca_extra[index], fmin=fmin, fmax=fmax,
                                                                 width=1.0, n_jobs=5)
                    else:
                        power = mne.time_frequency.tfr_stockwell(evoked_list[index], fmin=fmin, fmax=fmax, width=1.0,
                                                                 n_jobs=5)
                    power = power.pick_channels(channel)
                    # evoked_list.append(evoked)
                    power_list.append(power)
                if shorter_timescale:
                    tmin = -0.025
                    tmax = 0.065
                else:
                    tmin = -0.1
                    tmax = 0.3
                vmin = -380
                vmax = -280
                averaged = mne.grand_average(power_list, interpolate_bads=False, drop_bads=False)
                averaged.plot(channel, baseline=iv_baseline, mode='mean', cmap='jet',
                              axes=axes[count], show=False, colorbar=True, dB=True,
                              tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
                if count == 1:
                    axes[count].set_title('Time-Frequency Representations')
                im = axes[count].images
                cb = im[-1].colorbar
                # cb.set_label('dB', rotation=0, labelpad=15)
                cb.set_label('Amplitude (dB)')
                count += 1

                ##########################################################################################
                # Plot the topography
                ##########################################################################################
                evoked = mne.grand_average(evoked_list)
                # time_idx = []
                # tmp = np.argwhere(epochs.times >= time_point)
                # # sometimes when data is down sampled  find(epo.times == time_points(ii)) doesn't work
                # time_idx.append(tmp[0])
                # chanvalues = evoked.data[:, time_idx]
                # chan_labels = evoked.ch_names

                chanvalues = evoked.crop(tmin=time_point-(2/1000), tmax=time_point+(2/1000)).data
                chanvalues = chanvalues.mean(axis=1)  # Get average in window of interest

                chan_labels = evoked.ch_names
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

            # Add row titles
            rows = ['Uncleaned', 'PCA_OBS', 'ICA', 'SSP6']
            pad = 5
            for ax, row in zip(axes_unflat[:, 0], rows):
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', rotation=90)

            # Final Formatting
            plt.suptitle('Somatosensory Evoked Potentials', x=0.55)
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, top=0.95)
            if reduced_trials and shorter_timescale:
                fname = f"Combined_{cond_name}_reducedtrials_shorter"
            elif reduced_trials and not shorter_timescale:
                fname = f"Combined_{cond_name}_reducedtrials"
            elif shorter_timescale and not reduced_trials:
                fname = f"Combined_{cond_name}_shorter"
            else:
                fname = f"Combined_{cond_name}"

            plt.savefig(save_path+fname+'.png')
            # plt.savefig(save_path+fname+'.pdf')
            # plt.show()


