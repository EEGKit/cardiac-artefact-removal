# Run auto method of ICA on the prepared data as a comparison
# From MNE package - removes components with correlation coefficient greater than 0.9 with the ECG channel

import os
import mne
import matplotlib.pyplot as plt
from scipy.io import loadmat
from get_conditioninfo import *
from epoch_data import rereference_data
from Metrics.SNR_functions import evoked_from_raw


if __name__ == '__main__':
    subjects = [1, 6, 10]
    conditions = [2, 3]
    srmr_nr = 1
    sampling_rate = 1000

    run_ica = False
    plot_comparison = True
    reduced_trials = False

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

    image_path = "/data/p_02569/TestTan_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    if run_ica:
        for subject in subjects:
            for condition in conditions:
                # Set paths
                subject_id = f'sub-{str(subject).zfill(3)}'
                save_path = "../tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"  # Saving to baseline_ica_py
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"  # Taking prepared data
                figure_path = "/data/p_02569/baseline_ICA_images/" + subject_id + "/"
                cfg_path = "/data/pt_02569/"  # Contains important info about experiment
                os.makedirs(save_path, exist_ok=True)
                os.makedirs(figure_path, exist_ok=True)

                # Get the condition information based on the condition read in
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                nerve = cond_info.nerve

                # load cleaned ESG data
                fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # make a copy to filter
                raw_filtered = raw.copy().drop_channels(['ECG'])

                # filtering
                cfg = loadmat(cfg_path + 'cfg.mat')
                notch_freq = cfg['notch_freq'][0]
                esg_bp_freq = cfg['esg_bp_freq'][0]

                raw_filtered.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                                    iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                raw_filtered.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                # ICA
                ica = mne.preprocessing.ICA(n_components=len(raw_filtered.ch_names), max_iter='auto', random_state=97,
                                            method='picard')
                ica.fit(raw_filtered)

                raw.load_data()

                # Automatically choose ICA components
                ica.exclude = []
                # find which ICs match the ECG pattern
                ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG')
                ica.exclude = ecg_indices

                # Just for visualising
                ica.plot_overlay(raw.copy().drop_channels(['ECG']), exclude=ecg_indices, picks='eeg')
                print(ica.exclude)
                ica.plot_scores(ecg_scores)

                # Apply the ica we got from the filtered data onto the unfiltered raw
                ica.apply(raw)

                # add reference channel to data - average rereferencing
                mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                # Save raw data
                fname = 'testtan_baseline_ica_auto_' + cond_name + '.fif'
                raw.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    if plot_comparison:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            for condition in conditions:
                # Get the condition information based on the condition read in
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                nerve = cond_info.nerve

                if cond_name == 'tibial':
                    channel = ['L1']

                elif cond_name == 'median':
                    channel = ['SC6']

                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                evoked_og = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_trials)
                evoked_og.reorder_channels(esg_chans)

                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                fname = f"testtan_baseline_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                evoked_tan = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_trials)
                evoked_tan.reorder_channels(esg_chans)

                data_tan = evoked_tan.pick_channels(channel).get_data().reshape(-1)
                data_og = evoked_og.pick_channels(channel).get_data().reshape(-1)

                fig = plt.figure()
                plt.plot(evoked_tan.times, data_tan, label='Tanh')
                plt.plot(evoked_og.times, data_og, label='Original')
                plt.legend()
                plt.title(f"{subject_id}, {trigger_name}")
                plt.savefig(image_path+f"{subject_id}_{trigger_name}.png")
