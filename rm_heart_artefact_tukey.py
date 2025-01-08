# The same as rm_heart_artefact except the fitted artefact is multiplied by a Tukey window to avoid edge effects

import os
from scipy.io import loadmat
from scipy.signal import firls
from PCA_OBS_tukey import *
from get_conditioninfo import *
from get_channels import *


def rm_heart_artefact_tukey(subject, condition, srmr_nr, sampling_rate, pchip):
    # Set variables
    subject_id = f'sub-{str(subject).zfill(3)}'
    cond_info = get_conditioninfo(condition, srmr_nr)
    nblocks = cond_info.nblocks
    cond_name = cond_info.cond_name
    stimulation = cond_info.stimulation
    trigger_name = cond_info.trigger_name

    # Setting paths
    input_path = "/data/pt_02569/tmp_data/prepared_py/"+subject_id
    input_path_m = "/data/pt_02569/tmp_data/prepared/"+subject_id
    save_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/"+subject_id
    os.makedirs(save_path, exist_ok=True)

    figure_path = save_path

    # For debugging just test one channel of each
    debug_channel = ['S35']
    _, esg_chans, _ = get_channels(subject, False, False, srmr_nr)  # Ignoring ECG and EOG channels

    # Dyanmically set filename
    if pchip:
        fname = f"noStimart_sr{sampling_rate}_{cond_name}_withqrs_pchip"
    else:
        fname = f"noStimart_sr{sampling_rate}_{cond_name}_withqrs"
    raw = mne.io.read_raw_fif(input_path + fname + '.fif', preload=True)

    # Read .mat file with QRS events
    fname_m = f"raw_{sampling_rate}_spinal_{cond_name}"
    matdata = loadmat(input_path_m+fname_m+'.mat')
    QRSevents_m = matdata['QRSevents']

    # Run once with a single channel and debug_mode = True to get window information
    for ch in debug_channel:
        # set PCA_OBS input variables
        channelNames = ['S35', 'Iz', 'SC1', 'S3', 'SC6', 'S20', 'L1', 'L4']
        # these channels will be plotted(only for debugging / testing)

        if pchip:
            name = 'pca_chan_' + ch + '_pchip'
        else:
            name = 'pca_chan_'+ch
        # run PCA_OBS
        PCA_OBS_kwargs = dict(
            debug_mode=True, qrs=QRSevents_m, sr=sampling_rate,
            savename=save_path+name,
            ch_names=channelNames, sub_nr=subject_id,
            condition=cond_name, current_channel=ch
        )
        # Apply function modifies the data in raw in place
        raw.copy().apply_function(PCA_OBS_tukey, picks=[ch], **PCA_OBS_kwargs)

        # This information is the same for each channel - run through fitting once to get vals, add to all channels
        # Add annotations to each channel
        keywords = ['window_start_idx', 'window_end_idx']
        fn = f"{save_path}pca_chan_{ch}_{cond_name}_pca_info.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            window_start = infile[keywords[0]][()].reshape(-1)
            window_end = infile[keywords[1]][()].reshape(-1)

        onset = [x/sampling_rate for x in window_start]  # Divide by sampling rate to make times
        duration = np.repeat(0.0, len(window_start))
        description = ['fit_start'] * len(window_start)
        raw.annotations.append(onset, duration, description, ch_names=[esg_chans] * len(window_start))

        onset = [x/sampling_rate for x in window_end]
        duration = np.repeat(0.0, len(window_end))
        description = ['fit_end'] * len(window_end)
        raw.annotations.append(onset, duration, description, ch_names=[esg_chans]*len(window_end))

    # Then run parallel for all channels with n_jobs set and debug_mode = False
    # set PCA_OBS input variables
    channelNames = ['S35', 'Iz', 'SC1', 'S3', 'SC6', 'S20', 'L1', 'L4']
    # these channels will be plotted(only for debugging / testing)

    # run PCA_OBS
    # In this case ch_names, current_channel and savename are dummy vars - not necessary really
    PCA_OBS_kwargs = dict(
        debug_mode=False, qrs=QRSevents_m, sr=sampling_rate,
        savename=save_path + 'pca_chan',
        ch_names=channelNames, sub_nr=subject_id,
        condition=cond_name, current_channel=ch
    )

    # Apply function modifies the data in raw in place
    raw.apply_function(PCA_OBS_tukey, picks=esg_chans, **PCA_OBS_kwargs, n_jobs=len(esg_chans))

    # Save the new mne structure with the cleaned data
    if pchip:
        raw.save(os.path.join(save_path, f'data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif'), fmt='double',
                 overwrite=True)
    else:
        raw.save(os.path.join(save_path, f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'), fmt='double',
                 overwrite=True)
