# Taking the peak-peak value in the potential window of interest, and seeing how this varies across trials both before
# and after CCA - Prepared, PCA, Post-ICA and SSP 5&6
# Using the coefficient of variance as the metric


import mne
import numpy as np
import h5py
from scipy.io import loadmat
from SNR_functions import *
import pandas as pd
from invert import invert
from scipy.stats import variation
from epoch_data import rereference_data
import matplotlib.pyplot as plt


if __name__ == '__main__':
    calc_prepared = False
    calc_PCA = False
    calc_post_ICA = False
    calc_SSP = False
    cca_flag = True  # Compute SNR for final CCA corrected data

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

    # Contains information on which CCA component to pick
    xls = pd.ExcelFile('/data/p_02569/Components.xls')
    df = pd.read_excel(xls, 'Dataset 1')
    df.set_index('Subject', inplace=True)

    ################################### Prepared Calculations #################################
    if calc_prepared:
        class save_SNR():
            def __init__(self):
                pass


        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        var_med = np.zeros((len(subjects), 1))
        var_tib = np.zeros((len(subjects), 1))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    # Possibly change for dataset 2 - shorter time window for SNR
                    potential_window = [12 / 1000, 32 / 1000]
                    if not cca_flag:
                        channel = 'L1'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    potential_window = [8 / 1000, 18 / 1000]
                    if not cca_flag:
                        channel = 'SC6'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load epochs resulting from CCA on prepared
                if cca_flag:
                    input_path = "/data/pt_02569/tmp_data/prepared_py_cca/" + subject_id + "/esg/prepro/"
                    epochs = mne.read_epochs(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                                             , preload=True)
                    channel = df.loc[subject_id, f"Prep_{cond_name}"]
                    inv = df.loc[subject_id, f"Prep_{cond_name}_inv"]
                    if inv == 'inv' or inv == '!inv':
                        epochs.apply_function(invert, picks=channel)

                    # Now we have epochs for correct channel, want to find peak-peak in potential window for each epoch
                    epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                    data = epochs.get_data(picks=channel)  # n_epochs, n_channels, n_times
                    data = np.squeeze(data)  # Remove channel dimension as we only select one
                    peak_peak_amp = np.ptp(data, axis=1, keepdims=True)  # Returns peak-peak val of each epoch (2000, 1)
                    # Then get variance of these peak-peak vals
                    var = variation(peak_peak_amp, axis=0)[0]  # Just one number

                else:
                    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                    raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}.fif", preload=True)
                    # add reference channel to data
                    mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                    raw.set_eeg_reference(ref_channels='average')  # Perform rereferencing
                    raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                               iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                    raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                    # Create epochs
                    events, event_ids = mne.events_from_annotations(raw)
                    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                        baseline=tuple(iv_baseline), preload=True)
                    # Now we have epochs for correct channel, want to find peak-peak in potential window for each epoch
                    epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                    data = epochs.get_data(picks=channel)  # n_epochs, n_channels, n_times
                    data = np.squeeze(data)  # Remove channel dimension as we only select one
                    peak_peak_amp = np.ptp(data, axis=1, keepdims=True)  # Returns peak-peak val of each epoch (2000, 1)
                    # Then get variance of these peak-peak vals
                    var = variation(peak_peak_amp, axis=0)[0]  # Just one number

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    var_med[subject - 1, 0] = var
                elif cond_name == 'tibial':
                    var_tib[subject - 1, 0] = var

        # Save to file to compare to matlab - only for debugging
        savesnr.var_med = var_med
        savesnr.var_tib = var_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

        if cca_flag:
            fn = f"/data/pt_02569/tmp_data/prepared_py_cca/variance.h5"
        else:
            fn = f"/data/pt_02569/tmp_data/prepared_py/variance.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    ################################### PCA Calculations #################################
    if calc_PCA:
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        var_med = np.zeros((len(subjects), 1))
        var_tib = np.zeros((len(subjects), 1))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    # Possibly change for dataset 2 - shorter time window for SNR
                    potential_window = [12 / 1000, 32 / 1000]
                    if not cca_flag:
                        channel = 'L1'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    potential_window = [8 / 1000, 18 / 1000]
                    if not cca_flag:
                        channel = 'SC6'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load epochs resulting from CCA on prepared
                if cca_flag:
                    input_path = "/data/pt_02569/tmp_data/ecg_rm_py_cca/" + subject_id + "/esg/prepro/"
                    fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    channel = df.loc[subject_id, f"PCA_{cond_name}"]
                    inv = df.loc[subject_id, f"PCA_{cond_name}_inv"]
                    if inv == 'inv' or inv == '!inv':
                        epochs.apply_function(invert, picks=channel)

                    # Now we have epochs for correct channel, want to find peak-peak in potential window for each epoch
                    epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                    data = epochs.get_data(picks=channel)  # n_epochs, n_channels, n_times
                    data = np.squeeze(data)  # Remove channel dimension as we only select one
                    peak_peak_amp = np.ptp(data, axis=1, keepdims=True)  # Returns peak-peak val of each epoch (2000, 1)
                    # Then get variance of these peak-peak vals
                    var = variation(peak_peak_amp, axis=0)[0]  # Just one number

                else:
                    input_path = "/data/pt_02569/tmp_data/epoched_py/" + subject_id + "/esg/prepro/"
                    epochs = mne.read_epochs(f"{input_path}epo_clean_{cond_name}.fif", preload=True)
                    # Now we have epochs for correct channel, want to find peak-peak in potential window for each epoch
                    epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                    data = epochs.get_data(picks=channel)  # n_epochs, n_channels, n_times
                    data = np.squeeze(data)  # Remove channel dimension as we only select one
                    peak_peak_amp = np.ptp(data, axis=1, keepdims=True)  # Returns peak-peak val of each epoch (2000, 1)
                    # Then get variance of these peak-peak vals
                    var = variation(peak_peak_amp, axis=0)[0]  # Just one number

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    var_med[subject - 1, 0] = var
                elif cond_name == 'tibial':
                    var_tib[subject - 1, 0] = var

        # Save to file to compare to matlab - only for debugging
        savesnr.var_med = var_med
        savesnr.var_tib = var_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

        if cca_flag:
            fn = f"/data/pt_02569/tmp_data/ecg_rm_py_cca/variance.h5"
        else:
            fn = f"/data/pt_02569/tmp_data/ecg_rm_py/variance.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))


    ################################### Post-ICA Calculations #################################
    if calc_post_ICA:
        class save_SNR():
            def __init__(self):
                pass


        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        var_med = np.zeros((len(subjects), 1))
        var_tib = np.zeros((len(subjects), 1))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    # Possibly change for dataset 2 - shorter time window for SNR
                    potential_window = [12 / 1000, 32 / 1000]
                    if not cca_flag:
                        channel = 'L1'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    potential_window = [8 / 1000, 18 / 1000]
                    if not cca_flag:
                        channel = 'SC6'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load epochs resulting from CCA on prepared
                if cca_flag:
                    input_path = "/data/pt_02569/tmp_data/ica_py_cca/" + subject_id + "/esg/prepro/"
                    fname = f"clean_ica_auto_{cond_name}.fif"
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    channel = df.loc[subject_id, f"Post-ICA_{cond_name}"]
                    inv = df.loc[subject_id, f"Post-ICA_{cond_name}_inv"]
                    if inv == 'inv' or inv == '!inv':
                        epochs.apply_function(invert, picks=channel)

                    # Now we have epochs for correct channel, want to find peak-peak in potential window for each epoch
                    epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                    data = epochs.get_data(picks=channel)  # n_epochs, n_channels, n_times
                    data = np.squeeze(data)  # Remove channel dimension as we only select one
                    peak_peak_amp = np.ptp(data, axis=1, keepdims=True)  # Returns peak-peak val of each epoch (2000, 1)
                    # Then get variance of these peak-peak vals
                    var = variation(peak_peak_amp, axis=0)[0]  # Just one number

                else:
                    input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
                    raw = mne.io.read_raw_fif(f"{input_path}clean_ica_auto_{cond_name}.fif", preload=True)
                    # Create epochs
                    events, event_ids = mne.events_from_annotations(raw)
                    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                        baseline=tuple(iv_baseline), preload=True)
                    # Now we have epochs for correct channel, want to find peak-peak in potential window for each epoch
                    epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                    data = epochs.get_data(picks=channel)  # n_epochs, n_channels, n_times
                    data = np.squeeze(data)  # Remove channel dimension as we only select one
                    peak_peak_amp = np.ptp(data, axis=1, keepdims=True)  # Returns peak-peak val of each epoch (2000, 1)
                    # Then get variance of these peak-peak vals
                    var = variation(peak_peak_amp, axis=0)[0]  # Just one number

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    var_med[subject - 1, 0] = var
                elif cond_name == 'tibial':
                    var_tib[subject - 1, 0] = var

        # Save to file to compare to matlab - only for debugging
        savesnr.var_med = var_med
        savesnr.var_tib = var_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

        if cca_flag:
            fn = f"/data/pt_02569/tmp_data/ica_py_cca/variance.h5"
        else:
            fn = f"/data/pt_02569/tmp_data/ica_py/variance.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    ################################### SSP Calculations #################################
    if calc_SSP:
        class save_SNR():
            def __init__(self):
                pass


        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        var_med = np.zeros((len(subjects), len(np.arange(5, 7))))
        var_tib = np.zeros((len(subjects), len(np.arange(5, 7))))

        for n in np.arange(5, 7):
            for subject in subjects:
                for cond_name in cond_names:
                    if cond_name == 'tibial':
                        trigger_name = 'Tibial - Stimulation'
                        # Possibly change for dataset 2 - shorter time window for SNR
                        potential_window = [12 / 1000, 32 / 1000]
                        if not cca_flag:
                            channel = 'L1'
                    elif cond_name == 'median':
                        trigger_name = 'Median - Stimulation'
                        potential_window = [8 / 1000, 18 / 1000]
                        if not cca_flag:
                            channel = 'SC6'

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Want the SNR
                    # Load epochs resulting from CCA on prepared
                    if cca_flag:
                        input_path = f"/data/p_02569/SSP_cca/{subject_id}/{n} projections/"
                        epochs = mne.read_epochs(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                        channel = df.loc[subject_id, f"SSP{n}_{cond_name}"]
                        inv = df.loc[subject_id, f"SSP{n}_{cond_name}_inv"]
                        if inv == 'inv' or inv == '!inv':
                            epochs.apply_function(invert, picks=channel)

                        # Now we have epochs for correct channel, want to find peak-peak in potential window for each epoch
                        epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                        data = epochs.get_data(picks=channel)  # n_epochs, n_channels, n_times
                        data = np.squeeze(data)  # Remove channel dimension as we only select one
                        peak_peak_amp = np.ptp(data, axis=1,
                                               keepdims=True)  # Returns peak-peak val of each epoch (2000, 1)
                        # Then get variance of these peak-peak vals
                        var = variation(peak_peak_amp, axis=0)[0]  # Just one number

                    else:
                        input_path = "/data/p_02569/SSP/" + subject_id
                        savename = input_path + "/" + str(n) + " projections/"
                        raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")
                        # Create epochs
                        events, event_ids = mne.events_from_annotations(raw)
                        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                        epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                            baseline=tuple(iv_baseline), preload=True)
                        # Now we have epochs for correct channel, want to find peak-peak in potential window for each epoch
                        epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                        data = epochs.get_data(picks=channel)  # n_epochs, n_channels, n_times
                        data = np.squeeze(data)  # Remove channel dimension as we only select one
                        peak_peak_amp = np.ptp(data, axis=1,
                                               keepdims=True)  # Returns peak-peak val of each epoch (2000, 1)
                        # Then get variance of these peak-peak vals
                        var = variation(peak_peak_amp, axis=0)[0]  # Just one number

                    # Now have one snr related to each subject and condition
                    if cond_name == 'median':
                        var_med[subject - 1, n-5] = var
                    elif cond_name == 'tibial':
                        var_tib[subject - 1, n-5] = var

        # Save to file to compare to matlab - only for debugging
        savesnr.var_med = var_med
        savesnr.var_tib = var_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

        if cca_flag:
            fn = f"/data/p_02569/SSP_cca/variance.h5"
        else:
            fn = f"/data/p_02569/SSP/variance.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    ########################## Print to screen #################################
    keywords = ['var_med', 'var_tib']
    if cca_flag:
        input_paths = ["/data/pt_02569/tmp_data/prepared_py_cca/",
                       "/data/pt_02569/tmp_data/ecg_rm_py_cca/",
                       "/data/pt_02569/tmp_data/ica_py_cca/",
                       "/data/p_02569/SSP_cca/"]
    else:
        input_paths = ["/data/pt_02569/tmp_data/prepared_py/",
                       "/data/pt_02569/tmp_data/ecg_rm_py/",
                       "/data/pt_02569/tmp_data/ica_py/",
                       "/data/p_02569/SSP/"]

    names = ['Prepared', 'PCA', 'Post-ICA', 'SSP']

    if cca_flag:
        print("\n")
        for i in np.arange(0, 4):
            input_path = input_paths[i]
            name = names[i]
            fn = f"{input_path}variance.h5"
            # All have shape (24, 1) bar SSP which is (24, 16)
            with h5py.File(fn, "r") as infile:
                # Get the data
                var_med = infile[keywords[0]][()]
                var_tib = infile[keywords[1]][()]

            average_med = np.nanmean(var_med, axis=0)
            average_tib = np.nanmean(var_tib, axis=0)

            if name == 'SSP':
                for n in np.arange(0, 2):
                    print(f'Var {name} Median {n + 5}: {average_med[n]:.4f}')
                    print(f'Var {name} Tibial {n + 5}: {average_tib[n]:.4f}')
            else:
                print(f'Var {name} Median: {average_med[0]:.4f}')
                print(f'Var {name} Tibial: {average_tib[0]:.4f}')
    else:
        print("\n")
        for i in np.arange(0, len(input_paths)):
            input_path = input_paths[i]
            name = names[i]
            fn = f"{input_path}variance.h5"
            # All have shape (24, 1) bar SSP which is (24, 16)
            with h5py.File(fn, "r") as infile:
                # Get the data
                var_med = infile[keywords[0]][()]
                var_tib = infile[keywords[1]][()]

            average_med = np.nanmean(var_med, axis=0)
            average_tib = np.nanmean(var_tib, axis=0)

            if name == 'SSP':
                for n in np.arange(0, 2):
                    print(f'Var {name} Median {n + 5}: {average_med[n]:.4f}')
                    print(f'Var {name} Tibial {n + 5}: {average_tib[n]:.4f}')
            else:
                print(f'Var {name} Median: {average_med[0]:.4f}')
                print(f'Var {name} Tibial: {average_tib[0]:.4f}')


