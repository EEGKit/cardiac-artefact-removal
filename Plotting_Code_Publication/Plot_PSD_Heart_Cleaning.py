# Script to plot the power spectral density of the grand average data before and after different processes
# Shows the periodogram of the evoked data about spinal triggers

import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000
    full = True  # Do up to 2000Hz, otherwise up to 400Hz

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

    image_path = "/data/p_02569/PSD_Plots_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    methods = [True, True, True]  # Can't do ICA when cca_flag = True
    method_names = ['SSP6', 'PCA', 'ICA']  # Will treat SSP separately since there are multiple

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for i in np.arange(0, len(methods)):  # Methods Applied
        if methods[i]:  # Allows us to apply to only methods of interest
            method = method_names[i]

            for cond_name in cond_names:  # Conditions (median, tibial)
                evoked_list_cleaned = []
                evoked_list_uncleaned = []
                med_channel = ['SC6']
                tib_channel = ['L1']

                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'

                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                for subject in subjects:  # All subjects
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                    raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif", preload=True)
                    mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                    raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                               method='iir',
                               iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                    raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                    evoked.reorder_channels(esg_chans)
                    evoked_list_uncleaned.append(evoked)

                    if method == 'PCA':
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
                        evoked_list_cleaned.append(evoked)

                    elif method == 'ICA':
                        input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                        fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked_list_cleaned.append(evoked)

                    elif method == 'SSP6':
                        input_path = f"/data/p_02569/SSP/{subject_id}/6 projections/"
                        raw = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked_list_cleaned.append(evoked)

                averaged_cleaned = mne.grand_average(evoked_list_cleaned, interpolate_bads=False, drop_bads=False)
                averaged_uncleaned = mne.grand_average(evoked_list_uncleaned, interpolate_bads=False, drop_bads=False)
                if cond_name == 'median':
                    channel_clean = averaged_cleaned.copy().pick_channels(med_channel)
                    channel_unclean = averaged_uncleaned.copy().pick_channels(med_channel)

                elif cond_name == 'tibial':
                    channel_clean = averaged_cleaned.copy().pick_channels(tib_channel)
                    channel_unclean = averaged_uncleaned.copy().pick_channels(tib_channel)

                clean_data = np.mean(channel_clean.data[:, :], axis=0)
                unclean_data = np.mean(channel_unclean.data[:, :], axis=0)

                plt.figure()
                plt.psd(clean_data, NFFT=512, Fs=1000, label='Cleaned')
                plt.psd(unclean_data, NFFT=512, Fs=1000, label='Uncleaned')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('PSD (dB/Hz)')
                plt.xlim([0, 400])
                plt.ylim([-240, -130])
                plt.yticks(np.arange(-240, -130, 20.0))
                fname = f"{method}_{trigger_name}"
                plt.legend()

                plt.title(f"Method: {method}, Condition: {trigger_name}")
                plt.savefig(image_path+fname+".png")
                plt.savefig(image_path+fname+".pdf", bbox_inches='tight', format="pdf")
                plt.close()
