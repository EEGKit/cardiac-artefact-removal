import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

if __name__ == '__main__':
    freqs = np.arange(5., 250., 3.)
    fmin, fmax = freqs[[0, -1]]
    subjects = np.arange(1, 37)
    # subjects = [1, 2]
    cond_names = ['tibial', 'median']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    # Want 200ms before R-peak and 400ms after R-peak
    # Baseline is the 100ms period before the artefact occurs
    iv_baseline = [-300 / 1000, -200 / 1000]
    # Want 200ms before and 400ms after the R-peak in our epoch - need baseline outside this
    iv_epoch = [-400 / 1000, 600 / 1000]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/SAB_ECG_TimeCourse/"
    os.makedirs(image_path, exist_ok=True)

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for cond_name in cond_names:
        fig, axis = plt.subplots(1, 1, figsize=(12, 4))
        evoked_list_ecg = []
        evoked_list_spinal = []
        evoked_list_cort = []

        if cond_name == 'tibial':
            full_name = 'Tibial Nerve Stimulation'
            trigger_name = 'qrs'
            channel_ecg = ['ECG']
            channel_spinal = ['L1']
            channel_cortical = ['Cz']

        elif cond_name == 'median':
            full_name = 'Median Nerve Stimulation'
            trigger_name = 'qrs'
            channel_ecg = ['ECG']
            channel_spinal = ['SC6']
            channel_cortical = ['CP4']

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            # ECG
            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
            raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif",
                                      preload=True)

            # mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
            # CARDIAC
            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                       method='iir',
                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked_ECG = evoked.copy().pick_channels(channel_ecg)
            evoked_ECG = evoked_ECG.set_channel_types({'ECG': 'eeg'})  # Needed to do transform
            evoked_list_ecg.append(evoked_ECG)

            # CORTICAL
            evoked_cort = evoked.copy().pick_channels(channel_cortical)
            evoked_list_cort.append(evoked_cort)

            # SPINAL
            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
            raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif",
                                      preload=True)
            # mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                       method='iir',
                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked_spinal = evoked.copy().pick_channels(channel_spinal)
            evoked_list_spinal.append(evoked_spinal)
            # evoked_list.append(power)

        averaged_ecg = mne.grand_average(evoked_list_ecg, interpolate_bads=False, drop_bads=False)
        averaged_spinal = mne.grand_average(evoked_list_spinal, interpolate_bads=False, drop_bads=False)
        averaged_cortical = mne.grand_average(evoked_list_cort, interpolate_bads=False, drop_bads=False)
        plt.plot(averaged_ecg.times, averaged_ecg.data[0, :] * 10 ** 6, color='black')
        plt.plot(averaged_spinal.times, averaged_spinal.data[0, :] * 10 ** 6, color='black', linestyle='dashed')
        plt.plot(averaged_cortical.times, averaged_cortical.data[0, :] * 10 ** 6, color='grey', linestyle='dashed')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        plt.xlim([-0.3, 0.5])
        plt.ylabel('Amplitude (\u03BCV)')
        plt.xlabel('Time (s)')
        # plt.title(f'ECG Grand Average')
        plt.savefig(image_path+f"{full_name}")
        plt.savefig(image_path+f"{full_name}.pdf", bbox_inches='tight', format="pdf")
        plt.show()
