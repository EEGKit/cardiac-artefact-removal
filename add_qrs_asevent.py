# REDUNDANT SCRIPT NOW - INTEGRATED IN IMPORT_DATA.PY
# This script was later integrated in import_data.py - quicker to add separately originally
# Didn't know how to do it at the time, so instead taking the output from scripts import_data and rm_heart_artefact and
# adding the qrs peaks as events - will help with epoching around heart data later on
# Works specifically with the prepared_py and ecg_rm_py data - NOT GENERALISED TO ANY DATA

import numpy as np
import mne
import os
from scipy.io import loadmat
from get_conditioninfo import get_conditioninfo
from get_channels import get_channels


def add_qrs_asevent(subject, condition, srmr_nr, sampling_rate):
    # Set variables
    subject_id = f'sub-{str(subject).zfill(3)}'
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name

    # Read .mat file with QRS events
    input_path_m = "/data/pt_02569/tmp_data/prepared/"+subject_id+"/esg/prepro/"
    fname_m = f"raw_{sampling_rate}_spinal_{cond_name}"
    matdata = loadmat(input_path_m + fname_m + '.mat')
    QRSevents_m = matdata['QRSevents'][0]

    _, esg_chans, _ = get_channels(subject, False, False, srmr_nr)  # Ignoring ECG and EOG channels

    # Read in the raw data for prepared_py and add annotations, overwrite original to save
    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"  # Saving to prepared_py
    fname = f'noStimart_sr{sampling_rate}_{cond_name}.fif'
    raw = mne.io.read_raw_fif(input_path+fname)
    qrs_event = [x / sampling_rate for x in QRSevents_m]  # Divide by sampling rate to make times
    duration = np.repeat(0.0, len(QRSevents_m))
    description = ['qrs'] * len(QRSevents_m)
    raw.annotations.append(qrs_event, duration, description, ch_names=[esg_chans] * len(QRSevents_m))
    fname_save = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
    raw.save(os.path.join(input_path, fname_save), fmt='double', overwrite=True)

    # Read in the raw data for ecg_rm_py and add annotations, overwrite original to save
    # This one only needs to be run if annotations were not already added to the prepared data before initating heart
    # artefact removal
    input_path = "/data/pt_02569/tmp_data/ecg_rm_py/"+subject_id+"/esg/prepro/"  # Saving to prepared_py
    fname = f'data_clean_ecg_spinal_{cond_name}.fif'
    raw = mne.io.read_raw_fif(input_path+fname)
    qrs_event = [x / sampling_rate for x in QRSevents_m]  # Divide by sampling rate to make times
    duration = np.repeat(0.0, len(QRSevents_m))
    description = ['qrs'] * len(QRSevents_m)
    raw.annotations.append(qrs_event, duration, description, ch_names=[esg_chans] * len(QRSevents_m))
    fname_save = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
    raw.save(os.path.join(input_path, fname_save), fmt='double', overwrite=True)
