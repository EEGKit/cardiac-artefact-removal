# Run auto method of ICA on the prepared data anteriorly referenced
# From MNE package - removes components with correlation coefficient greater than 0.9 with the ECG channel

import os
import mne
import matplotlib.pyplot as plt
from scipy.io import loadmat
from get_conditioninfo import *
from epoch_data import rereference_data


def run_ica_anterior(subject, condition, srmr_nr, sampling_rate):

    # Set paths
    subject_id = f'sub-{str(subject).zfill(3)}'
    save_path = "../tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"  # Saving to baseline_ica_py
    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"  # Taking prepared data
    figure_path = "/data/p_02569/baseline_ICA_images/" + subject_id + "/"
    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    # Get the condition information based on the condition read in
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    nerve = cond_info.nerve

    # load cleaned ESG data
    fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # Anterior rereference data
    if nerve == 1:
        raw_antRef = rereference_data(raw, 'AC')
    elif nerve == 2:
        raw_antRef = rereference_data(raw, 'AL')

    # make a copy to filter
    raw_filtered = raw_antRef.copy().drop_channels(['ECG'])

    # filtering
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    raw_filtered.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    raw_filtered.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

    # ICA
    ica = mne.preprocessing.ICA(n_components=len(raw_filtered.ch_names), max_iter='auto', random_state=97)
    ica.fit(raw_filtered)

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

    raw_antRef.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw_antRef.ch_names), method='iir',
                      iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

    raw_antRef.notch_filter(freqs=notch_freq, n_jobs=len(raw_antRef.ch_names), method='fir', phase='zero')

    # Save data
    fname = 'anterior_clean_baseline_ica_auto_' + cond_name + '.fif'
    raw_antRef.save(os.path.join(save_path, fname), fmt='double', overwrite=True)
