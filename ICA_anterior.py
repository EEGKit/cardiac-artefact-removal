# Run auto method of ICA on the prepared data on anteriorly referenced data

import os
import mne
import matplotlib.pyplot as plt
from scipy.io import loadmat
from get_conditioninfo import *
from epoch_data import rereference_data


def run_ica_anterior(subject, condition, srmr_nr, sampling_rate):

    # Set paths
    subject_id = f'sub-{str(subject).zfill(3)}'
    save_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id   # Saving to baseline_ica_py
    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id   # Taking prepared data
    os.makedirs(save_path, exist_ok=True)

    # Get the condition information based on the condition read in
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    nerve = cond_info.nerve

    # load cleaned ESG data
    fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # Anterior re-reference data
    if nerve == 1:
        raw_antRef = rereference_data(raw, 'AC')
    elif nerve == 2:
        raw_antRef = rereference_data(raw, 'AL')

    # ICA
    ica = mne.preprocessing.ICA(n_components=len(raw_antRef.ch_names), max_iter='auto', random_state=97)
    ica.fit(raw_antRef)

    raw_antRef.load_data()

    # Automatically choose ICA components
    ica.exclude = []
    # find which ICs match the ECG pattern
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw_antRef, ch_name='ECG')
    ica.exclude = ecg_indices

    # Just for visualising
    # ica.plot_overlay(raw_antRef.copy().drop_channels(['ECG']), exclude=ecg_indices, picks='eeg')
    # print(ica.exclude)
    # ica.plot_scores(ecg_scores)

    # Apply the ica we got from the filtered data onto the unfiltered raw
    ica.apply(raw_antRef)

    # Save data
    fname = 'anterior_clean_baseline_ica_auto_' + cond_name + '.fif'
    raw_antRef.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    # Save ecg indices
    with open(f'anterior_ecg_indices_{cond_name}.txt', 'w') as file:
        file.write('\n'.join(str(ecg_index) for ecg_index in ecg_indices))

