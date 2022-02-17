# Script to actually run CCA on the data
# Using the meet package https://github.com/neurophysics/meet.git to run the CCA


import os
import mne
import h5py
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from get_conditioninfo import get_conditioninfo


def run_CCA(subject, condition, srmr_nr, data_string, n):
    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    subject_id = f'sub-{str(subject).zfill(3)}'

    # Select the right files based on the data_string
    if data_string == 'PCA':
        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
        fname = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
        save_path = "/data/pt_02569/tmp_data/ecg_rm_py_cca/" + subject_id + "/esg/prepro/"
        os.makedirs(save_path, exist_ok=True)

    elif data_string == 'ICA':
        input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
        fname = f'clean_baseline_ica_auto_{cond_name}.fif'
        save_path = "/data/pt_02569/tmp_data/baseline_ica_py_cca/" + subject_id + "/esg/prepro/"
        os.makedirs(save_path, exist_ok=True)

    elif data_string == 'Post-ICA':
        input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
        fname = f'clean_ica_auto_{cond_name}.fif'
        save_path = "/data/pt_02569/tmp_data/ica_py_cca/" + subject_id + "/esg/prepro/"
        os.makedirs(save_path, exist_ok=True)

    elif data_string == 'SSP':
        input_path = "/data/p_02569/SSP/" + subject_id + "/" + str(n) + " projections/"
        fname = f"ssp_cleaned_{cond_name}.fif"
        save_path = "/data/p_02569/SSP_cca/" + subject_id + "/" + str(n) + " projections/"
        os.makedirs(save_path, exist_ok=True)

    else:
        print('Invalid Data String Name Entered')
        exit()

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23', 'TH6']

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # PCA data has to be filtered before running CCA, all others have been filtered previously
    if data_string == 'PCA':
        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')


    # now create epochs based on the trigger names
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                        baseline=tuple(iv_baseline), preload=True)
    epochs = epochs.pick_channels(esg_chans, ordered=True)

    if cond_name == 'median':
        window_times = [8/1000, 18/1000]
        window = epochs.time_as_index(window_times)
    elif cond_name == 'tibial':
        window_times = [18/1000, 28/1000]
        window = epochs.time_as_index(window_times)
    else:
        print('Invalid condition name attempted for use')
        exit()

    # Crop the epochs
    epo_cca = epochs.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)

    # Prepare matrices for cca
    ##### Average matrix
    epo_av = epo_cca.copy().average().data.T
    # Now want channels x observations matrix #np.shape()[0] gets number of trials
    # Epo av is no_times x no_channels (10x40)
    # Want to repeat this to form an array thats no. observations x no.channels (20000x40)
    # Need to repeat the array, no_trials/times amount along the y axis
    avg_matrix = np.tile(epo_av, (int((np.shape(epochs.get_data())[0])), 1))
    avg_matrix = avg_matrix.T  # Need to transpose for correct form for function - channels x observations

    ##### Single trial matrix
    epo_cca_data = epo_cca.get_data(picks=esg_chans)
    epo_data = epochs.get_data(picks=esg_chans)

    # 0 to access number of epochs, 1 to access number of channels
    # channels x observations
    no_times = int(window[1] - window[0])
    # -1 means it automatically reshapes to what it needs after the other dimension is filled (with number of channels)
    # Need to transpose to get it in the form CCA wants
    st_matrix = np.swapaxes(epo_cca_data, 1, 2).reshape(-1, epo_cca_data.shape[1]).T
    st_matrix_long = np.swapaxes(epo_data, 1, 2).reshape(-1, epo_data.shape[1]).T

    # Run CCA
    W_avg, W_st, r = spatfilt.CCA_data(avg_matrix, st_matrix)  # Getting the same shapes as matlab so far

    if cond_name == 'median':
        W_st = W_st * -1  # Need to invert the weighting matrices to get the correct pattern, but not for tibial
    all_components = len(r)

    # Apply obtained weights to the long dataset (dimensions 40x9) - matrix multiplication
    CCA_concat = st_matrix_long.T @ W_st[:, 0:all_components]
    CCA_concat = CCA_concat.T

    # Spatial Patterns
    A_st = np.cov(st_matrix) @ W_st

    # Reshape - (900, 2000, 9)
    no_times_long = np.shape(epochs.get_data())[2]
    no_epochs = np.shape(epochs.get_data())[0]

    # Get into the same form as matlab and perform reshape as it does to be safe with reshape
    CCA_comps = np.reshape(CCA_concat, (all_components, no_times_long, no_epochs), order='F')

    # Now we have CCA comps, get the data in the axes format MNE likes (n_epochs, n_channels, n_times)
    CCA_comps = np.swapaxes(CCA_comps, 0, 2)
    CCA_comps = np.swapaxes(CCA_comps, 1, 2)
    selected_components = all_components  # Just keeping all for now to avoid rerunning

    # Need to create an epoch data class to store the information
    data = CCA_comps[:, 0:selected_components, :]
    events = epochs.events
    event_id = epochs.event_id
    tmin = iv_epoch[0]
    sfreq = 1000

    ch_names = []
    ch_types = []
    for i in np.arange(0, all_components):
        ch_names.append(f'Cor{i+1}')
        ch_types.append('eeg')

    # Initialize an info structure
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq
    )

    # Create and save
    cca_epochs = mne.EpochsArray(data, info, events, tmin, event_id)
    cca_epochs = cca_epochs.apply_baseline(baseline=tuple(iv_baseline))
    cca_epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)
