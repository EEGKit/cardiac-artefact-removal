# Plotting uncleaned and cleaned on top of one another to see fitted artefact from PCA - to see where it appears

import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, firls


if __name__ == '__main__':
    single_subject = False
    grand_average = True
    if single_subject:
        subjects = [2, 20]
    if grand_average:
        subjects = np.arange(1, 37)   # 1 through 36 to access subject data
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

    image_path = "/data/p_02569/FittedArtefact_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    if single_subject:
        for cond_name in cond_names:
            evoked_list_uncleaned = []
            evoked_list_PCA = []
            evoked_list_art = []

            # Changing to pick channel when each is loaded, not after the evoked list is formed
            if cond_name == 'tibial':
                trigger_name = 'Tibial - Stimulation'
                channel = ['L1']

            elif cond_name == 'median':
                trigger_name = 'Median - Stimulation'
                channel = ['SC6']

            for subject in subjects:
                subject_id = f'sub-{str(subject).zfill(3)}'

                for Method in ['Uncleaned', 'PCA_OBS', 'Artefact']:

                    # Same as above, but the label is different that's why I changed
                    if Method == 'Uncleaned':
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                        fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel, ordered=True)
                        evoked_list_uncleaned.append(evoked)

                    elif Method == 'PCA_OBS':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel, ordered=True)
                        evoked_list_PCA.append(evoked)

                    elif Method == 'Artefact':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                        fname = f"heart_artefact_{cond_name}_withqrs.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel, ordered=True)
                        evoked_list_art.append(evoked)

                # Get grand average
                averaged_prep = mne.grand_average(evoked_list_uncleaned, interpolate_bads=False, drop_bads=False)
                averaged_pca = mne.grand_average(evoked_list_PCA, interpolate_bads=False, drop_bads=False)
                averaged_art = mne.grand_average(evoked_list_art, interpolate_bads=False, drop_bads=False)

                plt.figure()
                plt.plot(averaged_prep.times, averaged_prep.data[0, :]*10**6, label='Uncleaned')
                plt.plot(averaged_pca.times, averaged_pca.data[0, :]*10**6, label='PCA_OBS')
                plt.plot(averaged_art.times, averaged_art.data[0, :]*10**6, label='Fitted Artefact')

                # plt.plot(evoked.times, np.mean(evoked.data[:, :], axis=0)*10**6,
                #          label=f'{Method}')

                plt.ylabel('Amplitude (\u03BCV)')
                plt.xlabel('Time (s)')
                plt.xlim([-0.05, 0.1])
                if cond_name == 'tibial':
                    plt.axvline(x=22 / 1000, color='k', linewidth=0.7, linestyle='dashed', label=None)
                    plt.title(f"Tibial Nerve Stimulation\n"
                              f"Lumbar Spinal Cord")
                elif cond_name == 'median':
                    plt.axvline(x=13 / 1000, color='k', linewidth=0.7, linestyle='dashed', label=None)
                    plt.title(f"Median Nerve Stimulation\n"
                              f"Cervical Spinal Cord")

                plt.axvline(x=0 / 1000, color='r', linewidth=0.7, linestyle='dashed', label=None)
                fname = f"{subject_id}_PCHIP_{cond_name}_pca_obs.png"
                plt.legend(loc='upper right')
                plt.savefig(image_path+fname)
                # plt.show()

    if grand_average:
        for cond_name in cond_names:
            evoked_list_uncleaned = []
            evoked_list_PCA = []
            evoked_list_art = []

            # Changing to pick channel when each is loaded, not after the evoked list is formed
            if cond_name == 'tibial':
                trigger_name = 'Tibial - Stimulation'
                channel = ['L1']

            elif cond_name == 'median':
                trigger_name = 'Median - Stimulation'
                channel = ['SC6']

            for subject in subjects:
                subject_id = f'sub-{str(subject).zfill(3)}'

                for Method in ['Uncleaned', 'PCA_OBS', 'Artefact']:

                    # Same as above, but the label is different that's why I changed
                    if Method == 'Uncleaned':
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                        fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel, ordered=True)
                        evoked_list_uncleaned.append(evoked)

                    elif Method == 'PCA_OBS':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel, ordered=True)
                        evoked_list_PCA.append(evoked)

                    elif Method == 'Artefact':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                        fname = f"heart_artefact_{cond_name}_withqrs.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                   method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel, ordered=True)
                        evoked_list_art.append(evoked)

            # Get grand average
            averaged_prep = mne.grand_average(evoked_list_uncleaned, interpolate_bads=False, drop_bads=False)
            averaged_pca = mne.grand_average(evoked_list_PCA, interpolate_bads=False, drop_bads=False)
            averaged_art = mne.grand_average(evoked_list_art, interpolate_bads=False, drop_bads=False)

            plt.figure()
            plt.plot(averaged_prep.times, averaged_prep.data[0, :] * 10 ** 6, label='Uncleaned')
            plt.plot(averaged_pca.times, averaged_pca.data[0, :] * 10 ** 6, label='PCA_OBS')
            plt.plot(averaged_art.times, averaged_art.data[0, :] * 10 ** 6, label='Fitted Artefact')

            # plt.plot(evoked.times, np.mean(evoked.data[:, :], axis=0)*10**6,
            #          label=f'{Method}')

            plt.ylabel('Amplitude (\u03BCV)')
            plt.xlabel('Time (s)')
            plt.xlim([-0.05, 0.1])
            if cond_name == 'tibial':
                plt.axvline(x=22 / 1000, color='k', linewidth=0.7, linestyle='dashed', label=None)
                plt.title(f"Tibial Nerve Stimulation\n"
                          f"Lumbar Spinal Cord")
            elif cond_name == 'median':
                plt.axvline(x=13 / 1000, color='k', linewidth=0.7, linestyle='dashed', label=None)
                plt.title(f"Median Nerve Stimulation\n"
                          f"Cervical Spinal Cord")

            plt.axvline(x=0 / 1000, color='r', linewidth=0.7, linestyle='dashed', label=None)
            fname = f"GrandAverage_PCHIP_{cond_name}_pca_obs.png"
            plt.legend(loc='upper right')
            plt.savefig(image_path + fname)
            # plt.show()
