# Compute ratio between the SEP amplitudes and the rms values of the artefact
import mne
import numpy as np
import h5py
import pandas as pd
from scipy.io import loadmat
from SNR_functions import *
from reref_data import rereference_data

if __name__ == '__main__':
    reduced_epochs = False
    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    # Loop through methods and save as required
    methods = ['Prep', 'PCA', 'ICA', 'SSP']

    df_med = pd.DataFrame(columns=methods)
    df_tib = pd.DataFrame(columns=methods)

    for method in methods:
        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    start = 12 / 1000
                    end = 32 / 1000
                    ch = 'L1'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    start = 8 / 1000
                    end = 18 / 1000
                    ch = 'SC6'

                subject_id = f'sub-{str(subject).zfill(3)}'

                if method == 'SSP':
                    # Load SSP projection data
                    file_path = "/data/pt_02569/tmp_data/ssp_py/"
                    file_name = f"{subject_id}/6 projections/ssp_cleaned_{cond_name}.fif"
                elif method == 'Prep':
                    file_path = f"/data/pt_02569/tmp_data/prepared_py/"
                    file_name = f'{subject_id}/noStimart_sr1000_{cond_name}_withqrs.fif'
                elif method == 'PCA':
                    file_path = f"/data/pt_02569/tmp_data/ecg_rm_py/"
                    file_name = f'{subject_id}/data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                elif method == 'ICA':
                    file_path = f"/data/pt_02569/tmp_data/baseline_ica_py/"
                    file_name = f'{subject_id}/clean_baseline_ica_auto_{cond_name}.fif'

                raw = mne.io.read_raw_fif(f"{file_path}{file_name}", preload=True)

                ##################################################################################
                # Get amplitude of SEP
                ##################################################################################
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                evoked_channel = evoked.copy().pick_channels([ch])

                # Extract relevant data in time of interest - to check if we even have negative value here
                time_idx = evoked_channel.time_as_index([start, end])
                data = evoked_channel.data[0, time_idx[0]:time_idx[1]]

                # Check data in channel actually has neg values - Extract negative peaks
                if np.any(data < 0):
                    _, latency, amplitude = evoked_channel.get_peak(ch_type=None, tmin=start, tmax=end, mode='neg',
                                                                    time_as_index=False, merge_grads=False,
                                                                    return_amplitude=True)
                    amplitude = abs(amplitude)
                # If there are no negative values, insert dummys
                else:
                    latency = np.nan
                    amplitude = np.nan

                iv_baseline_idx = evoked_channel.time_as_index([iv_baseline[0], iv_baseline[1]])
                base = evoked_channel.data[0, iv_baseline_idx[0]:iv_baseline_idx[1]]  # only one channel, baseline time period
                stan_dev = np.std(base, axis=0)

                # Compute snr
                snr = abs(amplitude) / abs(stan_dev)

                ################################################################################
                # Load rms value for this method type
                ################################################################################
                keywords = ['res_med', 'res_tib']
                esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                             'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                             'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                             'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                             'S23']
                median_pos = []
                tibial_pos = []
                for channel in ['L1']:
                    tibial_pos.append(esg_chans.index(channel))
                for channel in ['SC6']:
                    median_pos.append(esg_chans.index(channel))

                if method == 'SSP':
                    fn = f"/data/pt_02569/tmp_data/ssp_py/res_6.h5"
                else:
                    fn = f'{file_path}/res.h5'

                with h5py.File(fn, "r") as infile:
                    # Get the data
                    res_med = np.mean(infile[keywords[0]][()][:, median_pos])
                    res_tib = np.mean(infile[keywords[1]][()][:, tibial_pos])

                ###################################################################
                # Get ratio
                ###################################################################
                if cond_name == 'median':
                    ratio = snr/res_med
                    df_med.at[f'{subject-1}', f'{method}'] = ratio
                else:
                    ratio = snr/res_tib
                    df_tib.at[f'{subject-1}', f'{method}'] = ratio

    df_med = df_med.astype('float32')
    df_tib = df_tib.astype('float32')
    print('MEDIAN')
    print(df_med)
    print(df_med.describe())
    print('TIBIAL')
    print(df_tib)
    print(df_tib.describe())

    df_med.dropna(inplace=True)
    df_tib.dropna(inplace=True)
    print('MEDIAN')
    print(df_med)
    print(df_med.describe())
    print('TIBIAL')
    print(df_tib)
    print(df_tib.describe())