# Plotting uncleaned and cleaned on top of one another to see artefact

import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt

if __name__ == '__main__':
    group_avg = True
    single_trial = False
    # subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    # subjects = [2, 13, 20, 36]
    subjects = [2, 20]
    cond_names = ['median', 'tibial']
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

    image_path = "/data/p_02569/SingleSubjectComparison_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    if group_avg:
        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'
            for cond_name in cond_names:  # Conditions (median, tibial)
                plt.figure()
                # for Method in ['Uncleaned', 'Prepared', 'PCA', 'Artefact', 'PCA MATLAB', 'Prepared MAT + Py PCA']:
                for Method in ['Prepared', 'Prepared MAT', 'Prepared PCHIP Py', 'PCA', 'PCA MATLAB', 'PCA PCHIP',
                               'Prepared MAT + Py PCA']:
                # for Method in ['PCA', 'PCA Test', 'PCA MATLAB']:

                    # Changing to pick channel when each is loaded, not after the evoked list is formed
                    if cond_name == 'tibial':
                        trigger_name = 'Tibial - Stimulation'
                        channel = ['L1']

                    elif cond_name == 'median':
                        trigger_name = 'Median - Stimulation'
                        channel = ['SC6']

                    # raw data (no filtering)
                    # raw data with interpolated stimulus artifact  (no filtering)
                    # data after pca  (no filtering)
                    if Method == 'Uncleaned':
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                        fname = f'Stimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                    elif Method == 'Prepared':
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                        fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                    elif Method == 'Prepared PCHIP Py':
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                        fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs_pchip.fif'
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')


                    elif Method == 'PCA':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                    elif Method == 'PCA PCHIP':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                    elif Method == 'PCA Test':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs_test.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                    elif Method == 'Artefact':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                        fname = f"heart_artefact_{cond_name}_withqrs.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                    elif Method == 'PCA MATLAB':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm/" + subject_id + "/esg/prepro/"
                        fname = f"cnt_clean_ecg_spinal_{cond_name}.set"
                        raw = mne.io.read_raw_eeglab(input_path + fname, preload=True)
                        # mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                    elif Method == 'Prepared MAT + Py PCA':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs_mat.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        # mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                    elif Method == 'Prepared MATLAB':
                        input_path = "/data/pt_02569/tmp_data/prepared/" + subject_id + "/esg/prepro/"
                        fname = f"raw_1000_spinal_{cond_name}.set"
                        raw = mne.io.read_raw_eeglab(input_path + fname, preload=True)
                        # mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                    evoked.reorder_channels(esg_chans)
                    evoked = evoked.pick_channels(channel, ordered=True)

                    # relevant_channel = averaged.pick_channels(channel)
                    plt.plot(evoked.times, np.mean(evoked.data[:, :], axis=0)*10**6,
                             label=f'{Method}')

                plt.ylabel('Amplitude [\u03BCV]')
                plt.xlabel('Time [s]')
                # plt.xlim([-0.1, 0.3])
                plt.xlim([-0.05, 0.05])
                if cond_name == 'tibial':
                    plt.axvline(x=22 / 1000, color='r', linewidth=0.5, label='22ms')
                elif cond_name == 'median':
                    plt.axvline(x=13 / 1000, color='r', linewidth=0.5, label='13ms')
                plt.title(f"Subject {subject_id}, Condition: {trigger_name}")
                fname = f"{subject}_{trigger_name}_pcaprep.png"
                plt.legend(loc='upper right')
                # plt.savefig(image_path+fname)
                plt.show()
                plt.clf()

    if single_trial:
        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'
            for cond_name in cond_names:  # Conditions (median, tibial)
                for trial_no in [20, 501, 954, 1432, 1879]:
                    plt.figure()
                    # for Method in ['PCA', 'PCA MATLAB', 'Prepared MAT + Py PCA']:
                    for Method in ['Prepared', 'Prepared MATLAB']:
                        # Changing to pick channel when each is loaded, not after the evoked list is formed
                        if cond_name == 'tibial':
                            trigger_name = 'Tibial - Stimulation'
                            channel = ['L1']

                        elif cond_name == 'median':
                            trigger_name = 'Median - Stimulation'
                            channel = ['SC6']

                        # raw data (no filtering)
                        # raw data with interpolated stimulus artifact  (no filtering)
                        # data after pca  (no filtering)
                        if Method == 'PCA':
                            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                            fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                            mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                       method='iir',
                                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                        elif Method == 'PCA MATLAB':
                            input_path = "/data/pt_02569/tmp_data/ecg_rm/" + subject_id + "/esg/prepro/"
                            fname = f"cnt_clean_ecg_spinal_{cond_name}.set"
                            raw = mne.io.read_raw_eeglab(input_path + fname, preload=True)
                            # mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                       method='iir',
                                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                        elif Method == 'Prepared MAT + Py PCA':
                            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                            fname = f"data_clean_ecg_spinal_{cond_name}_withqrs_mat.fif"
                            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                            # mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                       method='iir',
                                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                        elif Method == 'Prepared':
                            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                            fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
                            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                            mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                       method='iir',
                                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                        elif Method == 'Prepared MATLAB':
                            input_path = "/data/pt_02569/tmp_data/prepared/" + subject_id + "/esg/prepro/"
                            fname = f"raw_1000_spinal_{cond_name}.set"
                            raw = mne.io.read_raw_eeglab(input_path + fname, preload=True)
                            # mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                       method='iir',
                                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                        events, event_ids = mne.events_from_annotations(raw)
                        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                        epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                            baseline=None, preload=True)
                        epochs = epochs.pick_channels(channel)
                        tstart = epochs.time_as_index(times=-0.1)[0]
                        tend = epochs.time_as_index(times=0.3)[0]
                        epochs = np.squeeze(epochs)  # Just 2000 trials and 901 timepoints now

                        # relevant_channel = averaged.pick_channels(channel)
                        # print(np.shape(np.linspace(-0.1, 0.3, num=901)))
                        # print(np.shape(epochs[trial_no, tstart:tend]))
                        plt.plot(np.linspace(-0.1, 0.3, num=400), epochs[trial_no, tstart:tend] * 10 ** 6,
                                 label=f'{Method}')

                    plt.ylabel('Amplitude [\u03BCV]')
                    plt.xlabel('Time [s]')
                    # plt.xlim([-0.1, 0.3])
                    plt.xlim([-0.05, 0.05])
                    if cond_name == 'tibial':
                        plt.axvline(x=22 / 1000, color='r', linewidth=0.5, label='22ms')
                    elif cond_name == 'median':
                        plt.axvline(x=13 / 1000, color='r', linewidth=0.5, label='13ms')
                    plt.title(f"Subject {subject_id}, Condition: {trigger_name}, Trial: {trial_no}")
                    fname = f"{subject}_{trigger_name}_{trial_no}_prep.png"
                    plt.legend(loc='upper right')
                    plt.savefig(image_path + fname)
                    plt.clf()
