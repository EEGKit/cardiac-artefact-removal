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
    load_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
    save_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id
    os.makedirs(save_path, exist_ok=True)

    # get condition info
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    nerve = cond_info.nerve

    # load dirty (prepared) ESG data
    fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'

    for n in np.arange(1, 21):
        raw = mne.io.read_raw_fif(load_path + fname, preload=True)

        # anterior reference
        if nerve == 1:
            raw_antRef = rereference_data(clean_raw, 'AC')
        elif nerve == 2:
            raw_antRef = rereference_data(clean_raw, 'AL')

        ############################################# SSP ##############################################
        # Normal data
        projs, events = mne.preprocessing.compute_proj_ecg(raw, n_eeg=n, reject=None, n_jobs=len(raw.ch_names),
                                                           ch_name='ECG')

        # Apply projections (clean data)
        clean_raw = raw.copy().add_proj(projs)
        clean_raw = clean_raw.apply_proj()

        # Anteriorly re-referenced data
        projs_ant, events_ant = mne.preprocessing.compute_proj_ecg(raw_antRef, n_eeg=n, reject=None,
                                                                   n_jobs=len(raw_antRef.ch_names), ch_name='ECG')

        # Apply projections (clean data)
        clean_raw_antRef = raw_antRef.copy().add_proj(projs_ant)
        clean_raw_antRef = clean_raw_antRef.apply_proj()

        ######################################### Save ##############################################
        savename = save_path + "/" + str(n) + " projections/"
        os.makedirs(savename, exist_ok=True)

        # Save the SSP cleaned data for future comparison
        clean_raw.save(f"{savename}ssp_cleaned_{cond_name}.fif", fmt='double', overwrite=True)
        clean_raw_antRef.save(f"{savename}ssp_cleaned_{cond_name}_antRef.fif", fmt='double', overwrite=True)
