# Compute SNR of the SSP cleaned data for each projection

# The SNR was estimated by dividing the ERP peak amplitude
# (absolute value) by the standard deviation of the LEP waveform in
# the pre-stimulus interval
# I'm gonna do one for positive going and one for negative going potentials
# Search for ERP as the max/min in the 0.005s after stimulus
# https://www.sciencedirect.com/science/article/abs/pii/S105381190901297X

import mne
import numpy as np
import h5py
from scipy.io import loadmat
from SNR_functions import *
from epoch_data import rereference_data
import matplotlib.pyplot as plt


if __name__ == '__main__':
    calc_prepared_snr = True
    calc_PCA_snr = True
    calc_post_ICA_snr = True
    calc_ICA_snr = True
    calc_SSP_snr = True
    reduced_epochs = True
    reduced_window = True  # Still need to SET WINDOW IN FUNCTION FILE FOR THIS TO WORK
    ant_ref = False  # Use the data that has been anteriorly referenced instead

    # Testing with just subject 1 at the moment
    subjects = np.arange(1, 37) # (1, 37) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    ################################### Prepared SNR Calculations #################################
    if calc_prepared_snr:
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 1))
        snr_tib = np.zeros((len(subjects), 1))
        chan_med = []
        chan_tib = []

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    nerve = 2
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    nerve = 1

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load epochs resulting from PCA OBS cleaning - the raw data in this folder has not been rereferenced
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}.fif", preload=True)

                # add reference channel to data
                if ant_ref:
                    # anterior reference
                    if nerve == 1:
                        raw = rereference_data(raw, 'AC')
                    elif nerve == 2:
                        raw = rereference_data(raw, 'AL')
                else:
                    mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                cfg_path = "/data/pt_02569/"  # Contains important info about experiment
                cfg = loadmat(cfg_path + 'cfg.mat')
                notch_freq = cfg['notch_freq'][0]
                esg_bp_freq = cfg['esg_bp_freq'][0]
                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                snr, chan = calculate_SNR_evoked(evoked, cond_name, iv_baseline)

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                    chan_med.append(chan)
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr
                    chan_tib.append(chan)

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        savesnr.chan_med = chan_med
        savesnr.chan_tib = chan_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if reduced_epochs:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/prepared_py/snr_reduced_ant_smallwin.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/prepared_py/snr_reduced_smallwin.h5"
            else:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/prepared_py/snr_ant_smallwin.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/prepared_py/snr_smallwin.h5"
        else:
            if reduced_epochs:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/prepared_py/snr_reduced_ant.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/prepared_py/snr_reduced.h5"
            else:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/prepared_py/snr_ant.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/prepared_py/snr.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

        # print(snr_med)
        # print(snr_tib)

    ################################### PCA SNR Calculations #################################
    if calc_PCA_snr:
        class save_SNR():
            def __init__(self):
                pass
        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 1))
        snr_tib = np.zeros((len(subjects), 1))
        chan_med = []
        chan_tib = []

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load epochs resulting from PCA OBS cleaning - has been rereferenced and filtered
                input_path = "/data/pt_02569/tmp_data/epoched_py/" + subject_id + "/esg/prepro/"
                if ant_ref:
                    epochs = mne.read_epochs(f"{input_path}epo_antRef_clean_{cond_name}.fif")
                else:
                    epochs = mne.read_epochs(f"{input_path}epo_clean_{cond_name}.fif")

                if reduced_epochs and trigger_name == 'Median - Stimulation':
                    epochs = epochs[900:1100]
                elif reduced_epochs and trigger_name == 'Tibial - Stimulation':
                    epochs = epochs[800:1200]

                evoked = epochs.average()

                snr, chan = calculate_SNR_evoked(evoked, cond_name, iv_baseline)

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                    chan_med.append(chan)
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr
                    chan_tib.append(chan)

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        savesnr.chan_med = chan_med
        savesnr.chan_tib = chan_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if reduced_epochs:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_reduced_ant_smallwin.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_reduced_smallwin.h5"
            else:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_ant_smallwin.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_smallwin.h5"
        else:
            if reduced_epochs:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_reduced_ant.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_reduced.h5"
            else:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_ant.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

        # print(snr_med)
        # print(snr_tib)


    ############################################## Post ICA SNR Calculations ########################################
    if calc_post_ICA_snr:
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 1))
        snr_tib = np.zeros((len(subjects), 1))
        chan_med = []
        chan_tib = []

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load epoched data resulting from post ICA cleaning
                input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
                if ant_ref:
                    raw = mne.io.read_raw_fif(f"{input_path}clean_ica_auto_antRef_{cond_name}.fif")
                else:
                    raw = mne.io.read_raw_fif(f"{input_path}clean_ica_auto_{cond_name}.fif")

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                snr, chan = calculate_SNR_evoked(evoked, cond_name, iv_baseline)

                # Now have an snr for each channel - want to average these to have one per subject and condition
                # Want to get average and stick it in correct matrix
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                    chan_med.append(chan)
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr
                    chan_tib.append(chan)

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        savesnr.chan_med = chan_med
        savesnr.chan_tib = chan_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

        if reduced_window:
            if reduced_epochs:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/ica_py/snr_reduced_ant_smallwin.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/ica_py/snr_reduced_smallwin.h5"
            else:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/ica_py/snr_ant_smallwin.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/ica_py/snr_smallwin.h5"
        else:
            if reduced_epochs:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/ica_py/snr_reduced_ant.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/ica_py/snr_reduced.h5"
            else:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/ica_py/snr_ant.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/ica_py/snr.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

        # print(snr_med)
        # print(snr_tib)

    ############################################### ICA alone SNR ######################################
    if calc_ICA_snr:
        class save_SNR():
            def __init__(self):
                pass
        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 1))
        snr_tib = np.zeros((len(subjects), 1))
        chan_med = []
        chan_tib = []

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load epoched data resulting from baseline ICA cleaning
                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                if ant_ref:
                    raw = mne.io.read_raw_fif(f"{input_path}clean_baseline_ica_auto_antRef_{cond_name}.fif")
                else:
                    raw = mne.io.read_raw_fif(f"{input_path}clean_baseline_ica_auto_{cond_name}.fif")

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                snr, chan = calculate_SNR_evoked(evoked, cond_name, iv_baseline)

                # Now have an snr for each channel - want to average these to have one per subject and condition
                # Want to get average and stick it in correct matrix
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                    chan_med.append(chan)
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr
                    chan_tib.append(chan)

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        savesnr.chan_med = chan_med
        savesnr.chan_tib = chan_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if reduced_epochs:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_reduced_ant_smallwin.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_reduced_smallwin.h5"
            else:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_ant_smallwin.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_smallwin.h5"
        else:
            if reduced_epochs:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_reduced_ant.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_reduced.h5"
            else:
                if ant_ref:
                    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_ant.h5"
                else:
                    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))
        # print(snr_med)
        # print(snr_tib)

    ############################### SSP Projectors SNR #################################################
    if calc_SSP_snr:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), len(np.arange(5, 21))))
        snr_tib = np.zeros((len(subjects), len(np.arange(5, 21))))
        # The channels will be saved in one long list - will be in order though so can reshape later
        chan_med = []
        chan_tib = []

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR for each projection tried from 5 to 20
                for n in np.arange(5, 21):  # (5, 21):
                    # Load SSP projection data
                    input_path = "/data/p_02569/SSP/" + subject_id
                    savename = input_path + "/" + str(n) + " projections/"
                    if ant_ref:
                        raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}_antRef.fif")
                    else:
                        raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")

                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                    snr, chan = calculate_SNR_evoked(evoked, cond_name, iv_baseline)

                    # Now have one snr for relevant channel in each subject + condition
                    if cond_name == 'median':
                        snr_med[subject - 1, n - 5] = snr
                        chan_med.append(chan)
                    elif cond_name == 'tibial':
                        snr_tib[subject - 1, n - 5] = snr
                        chan_tib.append(chan)

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        savesnr.chan_med = chan_med
        savesnr.chan_tib = chan_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if reduced_epochs:
                if ant_ref:
                    fn = f"/data/p_02569/SSP/snr_reduced_ant_smallwin.h5"
                else:
                    fn = f"/data/p_02569/SSP/snr_reduced_smallwin.h5"
            else:
                if ant_ref:
                    fn = f"/data/p_02569/SSP/snr_ant_smallwin.h5"
                else:
                    fn = f"/data/p_02569/SSP/snr_smallwin.h5"
        else:
            if reduced_epochs:
                if ant_ref:
                    fn = f"/data/p_02569/SSP/snr_reduced_ant.h5"
                else:
                    fn = f"/data/p_02569/SSP/snr_reduced.h5"
            else:
                if ant_ref:
                    fn = f"/data/p_02569/SSP/snr_ant.h5"
                else:
                    fn = f"/data/p_02569/SSP/snr.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

        # print(snr_med)
        # print(snr_tib)
        # print(chan_med)
        # print(chan_tib)
