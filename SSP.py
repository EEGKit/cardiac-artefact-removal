# Perform Signal Space Projection
# Mainly working from tutorial https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html
# SSP uses singular value decomposition to create the projection matrix

import os
import mne
from scipy.io import loadmat
from epoch_data import rereference_data
from get_conditioninfo import *
import numpy as np
import matplotlib.pyplot as plt


def apply_SSP(subject, condition, srmr_nr, sampling_rate):
    # set variables
    subject_id = f'sub-{str(subject).zfill(3)}'
    load_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    # save_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id + "/esg/prepro/"
    save_path = "/data/p_02569/SSP/" + subject_id
    os.makedirs(save_path, exist_ok=True)

    # get condition info
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    nerve = cond_info.nerve
    nblocks = cond_info.nblocks

    # Read relevant vars from cfg file
    # Both in ms - MNE works with seconds
    cfg = loadmat(cfg_path + 'cfg.mat')
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    # load dirty (prepared) ESG data
    fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'

    # for n in np.arange(5, 21):
    for n in np.arange(1, 21):
        raw = mne.io.read_raw_fif(load_path + fname, preload=True)

        ############################################# SSP ##############################################
        # create_ecg_epochs detects r-peaks and creates epochs around the r-wave peaks, capturing heartbeats
        # ecg_evoked = mne.preprocessing.create_ecg_epochs(raw, ch_name='ECG').average()
        # ecg_evoked.plot()

        # ecg_evoked.apply_baseline((None, None))
        # ecg_evoked.plot()

        # n = 10
        # Leaving everything default values
        projs, events = mne.preprocessing.compute_proj_ecg(raw, n_eeg=n, reject=None, n_jobs=len(raw.ch_names),
                                                           ch_name='ECG')

        # Apply projections (clean data)
        clean_raw = raw.copy().add_proj(projs)
        clean_raw = clean_raw.apply_proj()

        ########################### Re - Reference ####################################

        # Fz reference
        raw_FzRef = rereference_data(clean_raw, 'Fz-TH6')

        # anterior reference
        # if nerve == 1:
        #     raw_antRef = rereference_data(clean_raw, 'AC')
        # elif nerve == 2:
        #     raw_antRef = rereference_data(clean_raw, 'AL')

        # add reference channel to data - make sure recording reference is included
        mne.add_reference_channels(clean_raw, ref_channels=['TH6'], copy=False)  # Modifying in place

        ########################### Filtering ############################################

        clean_raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                         iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
        # raw_FzRef.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw_FzRef.ch_names), method='iir',
        #                  iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
        # raw_antRef.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw_antRef.ch_names), method='iir',
        #                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

        clean_raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
        # raw_FzRef.notch_filter(freqs=notch_freq, n_jobs=len(raw_FzRef.ch_names), method='fir', phase='zero')
        # raw_antRef.notch_filter(freqs=notch_freq, n_jobs=len(raw_antRef.ch_names), method='fir', phase='zero')

        ######################################### Plots ##############################################
        # Epoch around spinal triggers and plot
        # Need to do this with projections applied aka clean raw object
        events, event_ids = mne.events_from_annotations(clean_raw)
        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
        epochs = mne.Epochs(clean_raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                            baseline=tuple(iv_baseline))

        savename = save_path + "/" + str(n) + " projections/"
        os.makedirs(savename, exist_ok=True)

        # Save the SSP cleaned data for future comparison
        clean_raw.save(f"{savename}ssp_cleaned_{cond_name}.fif", fmt='double', overwrite=True)
        # raw_antRef.save(f"{savename}ssp_cleaned_{cond_name}_antRef.fif", fmt='double', overwrite=True)
        # raw_FzRef.save(f"{savename}ssp_cleaned_{cond_name}_FzRef.fif", fmt='double', overwrite=Tr
