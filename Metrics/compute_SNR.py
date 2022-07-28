# Compute SNR of the data for each method
# The SNR was estimated by dividing the ERP peak amplitude of the evoked response
# (absolute value) by the standard deviation of the LEP waveform in
# the pre-stimulus interval
# https://www.sciencedirect.com/science/article/abs/pii/S105381190901297X

import mne
import numpy as np
import h5py
from scipy.io import loadmat
from SNR_functions import *
from epoch_data import rereference_data

if __name__ == '__main__':
    reduced_epochs = False  # Use a smaller number of epochs to calculate the SNR
    reduced_window = False  # Smaller window about expected peak
    ant_ref = False  # Use the data that has been anteriorly referenced instead
    choose_limited = False  # Use ICA with limited components removed

    # Testing with just subject 1 at the moment
    subjects = np.arange(1, 37)  # (1, 37) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    # Loop through methods and save as required
    which_method = {'Prep': True,
                    'PCA': True,
                    'PCA PCHIP': True,
                    'PCA Tukey': True,
                    'PCA Tukey PCHIP': True,
                    'ICA': True,
                    'Post-ICA': True,
                    'SSP': True}

    for i in np.arange(0, len(which_method)):
        method = list(which_method.keys())[i]
        if which_method[method]:  # If this method is true, go through with the rest
            class save_SNR():
                def __init__(self):
                    pass

            # Instantiate class
            savesnr = save_SNR()

            # Treat SSP separately - has extra loop for projections
            if method == 'SSP':
                snr_med = np.zeros((len(subjects), len(np.arange(5, 21))))
                snr_tib = np.zeros((len(subjects), len(np.arange(5, 21))))
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
                        for n in np.arange(5, 21):
                            # Load SSP projection data
                            input_path = "/data/p_02569/SSP/" + subject_id
                            savename = input_path + "/" + str(n) + " projections/"
                            if ant_ref:
                                raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}_antRef.fif")
                            else:
                                raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")
                            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                            snr, chan = calculate_SNR_evoked(evoked, cond_name, iv_baseline, reduced_window)

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

            # All other methods
            else:
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

                        # Get the right file path
                        if method == 'Prep':
                            file_path = "/data/pt_02569/tmp_data/prepared_py/"
                            file_name = f'noStimart_sr1000_{cond_name}_withqrs.fif'
                        elif method == 'PCA':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                        elif method == 'PCA PCHIP':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif'
                        elif method == 'PCA Tukey':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                        elif method == 'PCA Tukey PCHIP':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif'
                        elif method == 'ICA':
                            file_path = "/data/pt_02569/tmp_data/baseline_ica_py/"
                            if choose_limited:
                                file_name = f'clean_baseline_ica_auto_{cond_name}_lim.fif'
                            else:
                                file_name = f'clean_baseline_ica_auto_{cond_name}.fif'
                        elif method == 'Post-ICA':
                            file_path = "/data/pt_02569/tmp_data/ica_py/"
                            file_name = f'clean_ica_auto_{cond_name}.fif'

                        input_path = file_path + subject_id + "/esg/prepro/"
                        raw = mne.io.read_raw_fif(f"{input_path}{file_name}", preload=True)

                        if (method == 'Prep' or method == 'PCA' or method == 'PCA Tukey' or method == 'PCA PCHIP' or
                                method == 'PCA Tukey PCHIP'):
                            # add reference channel to data
                            mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                       method='iir',
                                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                        if ant_ref:
                            # anterior reference
                            if nerve == 1:
                                raw = rereference_data(raw, 'AC')
                            elif nerve == 2:
                                raw = rereference_data(raw, 'AL')

                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                        snr, chan = calculate_SNR_evoked(evoked, cond_name, iv_baseline, reduced_window)

                        # Now have one snr related to each subject and condition
                        if cond_name == 'median':
                            snr_med[subject - 1, 0] = snr
                            chan_med.append(chan)
                        elif cond_name == 'tibial':
                            snr_tib[subject - 1, 0] = snr
                            chan_tib.append(chan)

                savesnr.snr_med = snr_med
                savesnr.snr_tib = snr_tib
                savesnr.chan_med = chan_med
                savesnr.chan_tib = chan_tib
                dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

                if reduced_window:
                    if reduced_epochs:
                        if ant_ref:
                            fn = f"/{file_path}snr_reducedtrials_ant_smallwin.h5"
                        else:
                            fn = f"{file_path}snr_reduced_smallwin.h5"
                    else:
                        if ant_ref:
                            fn = f"{file_path}snr_ant_smallwin.h5"
                        else:
                            fn = f"{file_path}snr_smallwin.h5"
                else:
                    if reduced_epochs:
                        if ant_ref:
                            fn = f"{file_path}snr_reduced_ant.h5"
                        else:
                            fn = f"{file_path}snr_reduced.h5"
                    else:
                        if ant_ref:
                            fn = f"{file_path}snr_ant.h5"
                        else:
                            fn = f"{file_path}snr.h5"
                if method == 'ICA' and choose_limited:
                    fn = f"{file_path}snr_lim.h5"
                if method == 'PCA PCHIP' or method == 'PCA Tukey PCHIP':
                    fn = f"{file_path}snr_pchip.h5"
                with h5py.File(fn, "w") as outfile:
                    for keyword in dataset_keywords:
                        outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    ############### Print to Screen numbers #################
    keywords = ['snr_med', 'snr_tib']
    input_paths = {'Prep': "/data/pt_02569/tmp_data/prepared_py/",
                   'PCA': "/data/pt_02569/tmp_data/ecg_rm_py/",
                   'PCA PCHIP': "/data/pt_02569/tmp_data/ecg_rm_py/",
                   'PCA Tukey': "/data/pt_02569/tmp_data/ecg_rm_py_tukey/",
                   'PCA Tukey PCHIP': "/data/pt_02569/tmp_data/ecg_rm_py_tukey/",
                   'ICA': "/data/pt_02569/tmp_data/baseline_ica_py/",
                   'Post-ICA': "/data/pt_02569/tmp_data/ica_py/",
                   'SSP': "/data/p_02569/SSP/"}

    print("\n")
    for i in np.arange(0, len(input_paths)):
        name = list(input_paths.keys())[i]
        input_path = input_paths[name]
        if name == 'ICA' and choose_limited:
            fn = f"{input_path}snr_lim.h5"
        elif name == 'PCA Tukey PCHIP' or name == 'PCA PCHIP':
            fn = f"{input_path}snr_pchip.h5"
        else:
            fn = f"{input_path}snr.h5"
        # All have shape (24, 1) bar SSP which is (36, 16)
        with h5py.File(fn, "r") as infile:
            # Get the data
            snr_med = infile[keywords[0]][()]
            snr_tib = infile[keywords[1]][()]

        average_med = np.nanmean(snr_med, axis=0)
        average_tib = np.nanmean(snr_tib, axis=0)

        if name == 'SSP':
            for n in np.arange(0, 16):
                print(f'SNR {name} Median {n + 5}: {average_med[n]:.4f}')
                print(f'SNR {name} Tibial {n + 5}: {average_tib[n]:.4f}')
        else:
            print(f'SNR {name} Median: {average_med[0]:.4f}')
            print(f'SNR {name} Tibial: {average_tib[0]:.4f}')

