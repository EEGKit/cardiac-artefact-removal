# Taking the peak-peak value in the potential window of interest, and seeing how this varies across trials both before
# and after CCA - Prepared, PCA, and SSP 5&6
# Using the coefficient of variation as the metric


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


def get_cropped_epochs(raw_data, epoch_interval, baseline_interval, interest_window, trigger):
    es, e_ids = mne.events_from_annotations(raw_data)  # Get events and ids
    e_id_dict = {key: value for key, value in e_ids.items() if key == trigger}  # Extract relevant
    epos = mne.Epochs(raw_data, es, event_id=e_id_dict, tmin=epoch_interval[0], tmax=epoch_interval[1],
                      baseline=tuple(baseline_interval), preload=True)  # Create epochs
    epos = epos.crop(tmin=interest_window[0], tmax=interest_window[1])  # Crop

    return epos

# Controlling peak-peak such that positivity must be before negativity
def get_coeffofvariation(epos, ch):
    data = epos.get_data(picks=ch)  # n_epochs, n_channels, n_times
    data = np.squeeze(data)  # Remove channel dimension as we only have one, shape (2000, 11)
    # Loop through each epoch
    ptp = []
    count=0
    for row in data:
        min = np.min(row)
        idx = np.argmin(row)
        # truncate the epoch so we only keep values before occurrence of minimum
        # If minimum if in the first position of the time window, the ptp will be nan
        if idx == 0:
            ptp.append(np.nan)
            count += 1
        else:
            row_max = row[0:idx]
            max = np.max(row_max)
            ptp.append(max-min)
    # Then get variance of these peak-peak vals
    coeff_var = variation(ptp, axis=0, nan_policy='omit')  # Just one number

    return count, coeff_var


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    which_method = {'Prep': True,
                    'PCA': True,
                    'SSP': True}
    cca_flag = False  # Compute for CCA corrected data (True) or normal (False)

    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    # Contains information on which CCA component to pick - manually selected previously
    xls = pd.ExcelFile('/data/p_02569/Components.xls')
    df = pd.read_excel(xls, 'Dataset 1')
    df.set_index('Subject', inplace=True)

    df_median = pd.DataFrame(columns=['Prep', 'PCA', 'SSP'])
    df_tibial = pd.DataFrame(columns=['Prep', 'PCA', 'SSP'])
    # for df_setup in [df_median, df_tibial]:
    #     df_setup['Subjects'] = np.arange(1, 37)
        # df_setup.set_index('Subjects', inplace=True)

    for i in np.arange(0, len(which_method)):
        method = list(which_method.keys())[i]
        if which_method[method]:  # If this method is true, go through with the rest
            class save_Var():
                def __init__(self):
                    pass

            # Instantiate class
            savevar = save_Var()

            if method == 'SSP':
                # Matrix of dimensions no.subjects x no. projections
                var_med = np.zeros((len(subjects), len(np.arange(5, 7))))
                var_tib = np.zeros((len(subjects), len(np.arange(5, 7))))
            else:
                # Matrix of dimensions no.subjects
                var_med = np.zeros((len(subjects), 1))
                var_tib = np.zeros((len(subjects), 1))

            # For cca corrected data
            if cca_flag:
                for subject in subjects:
                    for cond_name in cond_names:
                        # Get the correct condition variables
                        if cond_name == 'tibial':
                            trigger_name = 'Tibial - Stimulation'
                            potential_window = [12 / 1000, 32 / 1000]
                        elif cond_name == 'median':
                            trigger_name = 'Median - Stimulation'
                            potential_window = [8 / 1000, 18 / 1000]

                        subject_id = f'sub-{str(subject).zfill(3)}'

                        # Get the right file path
                        if method == 'Prep':
                            file_path = "/data/pt_02569/tmp_data/prepared_py_cca/"
                            file_name = f'noStimart_sr1000_{cond_name}_withqrs.fif'
                        elif method == 'PCA':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py_cca/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                        elif method == 'SSP':
                            file_path = "/data/pt_02569/tmp_data/ssp_py_cca/"
                            file_name = f'ssp_cleaned_{cond_name}.fif'

                        # Process SSP data
                        if method == 'SSP':
                            for n in np.arange(5, 7):
                                input_path = file_path + subject_id + "/" + str(n) + " projections/"
                                epochs = mne.read_epochs(f"{input_path}{file_name}", preload=True)
                                channel = df.loc[subject_id, f"{method}{n}_{cond_name}"]
                                inv = df.loc[subject_id, f"{method}{n}_{cond_name}_inv"]
                                if inv == 'inv' or inv == '!inv':
                                    epochs.apply_function(invert, picks=channel)

                                epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                                no_drops, var = get_coeffofvariation(epochs, channel)
                                if cond_name == 'median':
                                    var_med[subject - 1, n - 5] = var
                                    if n == 6:
                                        df_median.at[f'{subject-1}', f'{method}'] = no_drops
                                elif cond_name == 'tibial':
                                    var_tib[subject - 1, n - 5] = var
                                    if n == 6:
                                        df_tibial.at[f'{subject-1}', f'{method}'] = no_drops

                        # Process all other methods
                        else:
                            input_path = file_path + subject_id
                            epochs = mne.read_epochs(f"{input_path}{file_name}", preload=True)
                            channel = df.loc[subject_id, f"{method}_{cond_name}"]
                            inv = df.loc[subject_id, f"{method}_{cond_name}_inv"]
                            if inv == 'inv' or inv == '!inv':
                                epochs.apply_function(invert, picks=channel)

                            epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                            no_drops, var = get_coeffofvariation(epochs, channel)
                            if cond_name == 'median':
                                var_med[subject - 1, 0] = var
                                df_median.at[f'{subject-1}', f'{method}'] = no_drops
                            elif cond_name == 'tibial':
                                var_tib[subject - 1, 0] = var
                                df_tibial.at[f'{subject-1}', f'{method}'] = no_drops

                # Save to file
                savevar.var_med = var_med
                savevar.var_tib = var_tib
                dataset_keywords = [a for a in dir(savevar) if not a.startswith('__')]

                fn = f"{file_path}variance_controlptp.h5"
                with h5py.File(fn, "w") as outfile:
                    for keyword in dataset_keywords:
                        outfile.create_dataset(keyword, data=getattr(savevar, keyword))

            # Deal with non-cca corrected data
            else:
                for subject in subjects:
                    for cond_name in cond_names:
                        # Get correct condition information
                        if cond_name == 'tibial':
                            trigger_name = 'Tibial - Stimulation'
                            potential_window = [12 / 1000, 32 / 1000]
                            channel = 'L1'
                        elif cond_name == 'median':
                            trigger_name = 'Median - Stimulation'
                            potential_window = [8 / 1000, 18 / 1000]
                            channel = 'SC6'

                        subject_id = f'sub-{str(subject).zfill(3)}'

                        # Get the right file path
                        if method == 'Prep':
                            file_path = "/data/pt_02569/tmp_data/prepared_py/"
                            file_name = f'noStimart_sr1000_{cond_name}_withqrs.fif'
                        elif method == 'PCA':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                        elif method == 'SSP':
                            file_path = "/data/pt_02569/tmp_data/ssp_py/"
                            file_name = f'ssp_cleaned_{cond_name}.fif'

                        # Process SSP data
                        if method == 'SSP':
                            for n in np.arange(5, 7):
                                input_path = file_path + subject_id + "/" + str(n) + " projections/"
                                raw = mne.io.read_raw_fif(f"{input_path}{file_name}", preload=True)
                                # Create epochs
                                epochs = get_cropped_epochs(raw, iv_epoch, iv_baseline, potential_window, trigger_name)
                                no_drops, var = get_coeffofvariation(epochs, channel)
                                # Now have one snr related to each subject and condition
                                if cond_name == 'median':
                                    var_med[subject - 1, n-5] = var
                                    if n == 6:
                                        df_median.at[f'{subject-1}', f'{method}'] = no_drops
                                elif cond_name == 'tibial':
                                    var_tib[subject - 1, n-5] = var
                                    if n == 6:
                                        df_tibial.at[f'{subject-1}', f'{method}'] = no_drops

                        # Process all other methods
                        else:
                            input_path = file_path + subject_id
                            raw = mne.io.read_raw_fif(f"{input_path}{file_name}", preload=True)

                            epochs = get_cropped_epochs(raw, iv_epoch, iv_baseline, potential_window, trigger_name)
                            no_drops, var = get_coeffofvariation(epochs, channel)
                            if cond_name == 'median':
                                var_med[subject - 1, 0] = var
                                df_median.at[f'{subject-1}', f'{method}'] = no_drops
                            elif cond_name == 'tibial':
                                var_tib[subject - 1, 0] = var
                                df_tibial.at[f'{subject-1}', f'{method}'] = no_drops

                # Save to file
                savevar.var_med = var_med
                savevar.var_tib = var_tib
                dataset_keywords = [a for a in dir(savevar) if not a.startswith('__')]

                fn = f"{file_path}variance_controlptp.h5"
                with h5py.File(fn, "w") as outfile:
                    for keyword in dataset_keywords:
                        outfile.create_dataset(keyword, data=getattr(savevar, keyword))


    ########################## Print to screen #################################
    df_median = df_median.astype('float32')
    df_tibial= df_tibial.astype('float32')
    keywords = ['var_med', 'var_tib']
    if cca_flag:
        input_paths = ["/data/pt_02569/tmp_data/prepared_py_cca/",
                       "/data/pt_02569/tmp_data/ecg_rm_py_cca/",
                       "/data/pt_02569/tmp_data/ssp_py_cca/"]
    else:
        input_paths = ["/data/pt_02569/tmp_data/prepared_py/",
                       "/data/pt_02569/tmp_data/ecg_rm_py/",
                       "/data/pt_02569/tmp_data/ssp_py/"]

    names = ['Prepared', 'PCA', 'SSP']

    if cca_flag:
        print("\n")
        print('CCA Corrected Results')
        for i in np.arange(0, len(input_paths)):
            input_path = input_paths[i]
            name = names[i]
            fn = f"{input_path}variance_controlptp.h5"
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
        print('DROPS AFTER CCA')
        print('median')
        print(df_median)
        print(df_median.describe())
        print('tibial')
        print(df_tibial)
        print(df_tibial.describe())
    else:
        print("\n")
        print('Prior to CCA Results')
        for i in np.arange(0, len(input_paths)):
            input_path = input_paths[i]
            name = names[i]
            fn = f"{input_path}variance_controlptp.h5"
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
        print('DROPS BEFORE CCA')
        print('median')
        print(df_median)
        print(df_median.describe())
        print('tibial')
        print(df_tibial)
        print(df_tibial.describe())
