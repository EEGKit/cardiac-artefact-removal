# Script to convert raw objects to eeglab structures so I can run isopotential scripts

import mne
import numpy as np
from get_conditioninfo import get_conditioninfo

n_subjects = 36  # Number of subjects
# Testing with just subject 1 at the moment
# Take an assortment of subjects for now just to see evolution
# subjects = np.arange(1, 2)  # (1, 37) # 1 through 36 to access subject data
subjects = [1, 6, 11, 15, 22, 30]
srmr_nr = 1  # Experiment Number
conditions = [2, 3]  # Conditions of interest
sampling_rate = 1000

# Subject 1 PCA data
for subject in subjects:
    for condition in conditions:
        subject_id = f'sub-{str(subject).zfill(3)}'
        # Get the condition information based on the condition read in
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name

        input_path = f"/data/pt_02569/tmp_data/ecg_rm_py/{subject_id}/esg/prepro/"
        fname = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
        raw = mne.io.read_raw_fif(input_path+fname)

        fname = f'data_clean_ecg_spinal_{cond_name}_withqrs.set'
        raw.export(input_path+fname, fmt='eeglab', verbose=None)

# Subject 1 post_ICA data
for subject in subjects:
    for condition in conditions:
        subject_id = f'sub-{str(subject).zfill(3)}'
        # Get the condition information based on the condition read in
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name

        input_path = f"/data/pt_02569/tmp_data/ica_py/{subject_id}/esg/prepro/"
        fname = f'clean_ica_auto_{cond_name}.fif'
        raw = mne.io.read_raw_fif(input_path+fname)

        fname = f'clean_ica_auto_{cond_name}.set'
        raw.export(input_path+fname, fmt='eeglab', verbose=None)

# Baseline ICA data
for subject in subjects:
    for condition in conditions:
        subject_id = f'sub-{str(subject).zfill(3)}'
        # Get the condition information based on the condition read in
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name

        input_path = f"/data/pt_02569/tmp_data/baseline_ica_py/{subject_id}/esg/prepro/"
        fname = f'clean_baseline_ica_auto_{cond_name}.fif'
        raw = mne.io.read_raw_fif(input_path+fname)

        fname = f'clean_baseline_ica_auto_{cond_name}.set'
        raw.export(input_path+fname, fmt='eeglab', verbose=None)

# SSP data
for subject in subjects:
    for condition in conditions:
        for n in np.arange(5, 21):
            subject_id = f'sub-{str(subject).zfill(3)}'
            # Get the condition information based on the condition read in
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name

            input_path = f"/data/p_02569/SSP/{subject_id}/{str(n)} projections/"
            fname = f'ssp_cleaned_{cond_name}.fif'
            raw = mne.io.read_raw_fif(input_path + fname)

            fname = f'ssp_cleaned_{cond_name}.set'
            raw.export(input_path + fname, fmt='eeglab', verbose=None)