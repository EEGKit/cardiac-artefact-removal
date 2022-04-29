# Taking the peak-peak value in the potential window of interest, and seeing how this varies across trials both before
# and after CCA - Prepared, PCA, Post-ICA and SSP 5&6
# Using the coefficient of variance as the metric
# Computing for CCA with 1000, 500 and 250 trials used
# Using the same components to compute as are used for the 2000 trials version

import h5py
from scipy.io import loadmat
from SNR_functions import *
import pandas as pd
from invert import invert
from scipy.stats import variation

if __name__ == '__main__':
    calc_prepared = True
    calc_PCA = True
    calc_post_ICA = True
    calc_SSP = True

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
        for no in [1000, 500, 250]:
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

                    elif cond_name == 'median':
                        trigger_name = 'Median - Stimulation'
                        potential_window = [8 / 1000, 18 / 1000]

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Want the SNR
                    # Load epochs resulting from CCA on prepared
                    input_path = "/data/pt_02569/tmp_data/prepared_py_cca/" + subject_id + "/esg/prepro/"
                    epochs = mne.read_epochs(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs{no}.fif"
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

                    # Now have one snr related to each subject and condition
                    if cond_name == 'median':
                        var_med[subject - 1, 0] = var
                    elif cond_name == 'tibial':
                        var_tib[subject - 1, 0] = var

            # Save to file to compare to matlab - only for debugging
            savesnr.var_med = var_med
            savesnr.var_tib = var_tib
            dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

            fn = f"/data/pt_02569/tmp_data/prepared_py_cca/variance{no}.h5"
            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    ################################### PCA Calculations #################################
    if calc_PCA:
        for no in [1000, 500, 250]:
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

                    elif cond_name == 'median':
                        trigger_name = 'Median - Stimulation'
                        potential_window = [8 / 1000, 18 / 1000]

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Want the SNR
                    # Load epochs resulting from CCA on prepared
                    input_path = "/data/pt_02569/tmp_data/ecg_rm_py_cca/" + subject_id + "/esg/prepro/"
                    fname = f"data_clean_ecg_spinal_{cond_name}_withqrs{no}.fif"
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

                    # Now have one snr related to each subject and condition
                    if cond_name == 'median':
                        var_med[subject - 1, 0] = var
                    elif cond_name == 'tibial':
                        var_tib[subject - 1, 0] = var

            # Save to file to compare to matlab - only for debugging
            savesnr.var_med = var_med
            savesnr.var_tib = var_tib
            dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

            fn = f"/data/pt_02569/tmp_data/ecg_rm_py_cca/variance{no}.h5"
            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savesnr, keyword))


    ################################### Post-ICA Calculations #################################
    if calc_post_ICA:
        for no in [1000, 500, 250]:
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

                    elif cond_name == 'median':
                        trigger_name = 'Median - Stimulation'
                        potential_window = [8 / 1000, 18 / 1000]

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Want the SNR
                    # Load epochs resulting from CCA on prepared
                    input_path = "/data/pt_02569/tmp_data/ica_py_cca/" + subject_id + "/esg/prepro/"
                    fname = f"clean_ica_auto_{cond_name}{no}.fif"
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

                    # Now have one snr related to each subject and condition
                    if cond_name == 'median':
                        var_med[subject - 1, 0] = var
                    elif cond_name == 'tibial':
                        var_tib[subject - 1, 0] = var

            # Save to file to compare to matlab - only for debugging
            savesnr.var_med = var_med
            savesnr.var_tib = var_tib
            dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

            fn = f"/data/pt_02569/tmp_data/ica_py_cca/variance{no}.h5"
            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    ################################### SSP Calculations #################################
    if calc_SSP:
        for no in [1000, 500, 250]:
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

                        elif cond_name == 'median':
                            trigger_name = 'Median - Stimulation'
                            potential_window = [8 / 1000, 18 / 1000]

                        subject_id = f'sub-{str(subject).zfill(3)}'

                        # Want the SNR
                        # Load epochs resulting from CCA on SSP
                        input_path = f"/data/p_02569/SSP_cca/{subject_id}/{n} projections/"
                        epochs = mne.read_epochs(f"{input_path}ssp_cleaned_{cond_name}{no}.fif", preload=True)
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

                        # Now have one snr related to each subject and condition
                        if cond_name == 'median':
                            var_med[subject - 1, n-5] = var
                        elif cond_name == 'tibial':
                            var_tib[subject - 1, n-5] = var

            # Save to file to compare to matlab - only for debugging
            savesnr.var_med = var_med
            savesnr.var_tib = var_tib
            dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

            fn = f"/data/p_02569/SSP_cca/variance{no}.h5"
            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    ########################## Print to screen #################################
    keywords = ['var_med', 'var_tib']
    input_paths = ["/data/pt_02569/tmp_data/prepared_py_cca/",
                   "/data/pt_02569/tmp_data/ecg_rm_py_cca/",
                   "/data/pt_02569/tmp_data/ica_py_cca/",
                   "/data/p_02569/SSP_cca/"]

    names = ['Prepared', 'PCA', 'Post-ICA', 'SSP']

    for no in [1000, 500, 250]:
        print("\n")
        for i in np.arange(0, 4):
            input_path = input_paths[i]
            name = names[i]
            fn = f"{input_path}variance{no}.h5"
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
