# Run auto method of ICA on the prepared data separately for lumbar and cervical patches
# From MNE package - removes components with correlation coefficient greater than 0.9 with the ECG channel

import os
import mne
import matplotlib.pyplot as plt
from scipy.io import loadmat
from get_conditioninfo import *
from epoch_data import rereference_data
from get_esg_channels import get_esg_channels


def run_ica_separatepatches(subject, condition, srmr_nr, sampling_rate):

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

    # Get respective channel types
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()
    # load cleaned ESG data
    fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # Make a filtered copy
    raw_filtered = raw.copy().drop_channels(['ECG'])
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    raw_filtered.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    raw_filtered.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

    # ICA
    # Perform ica separately for lumbar and cervical channels
    # Only process one patch for median or tibial and ignore for the rest of analysis
    if cond_name == 'median':
        channels = cervical_chans
    elif cond_name == 'tibial':
        channels = lumbar_chans

    channels.append('ECG')  # Needed as we use ECG to find bad channels

    raw_patch = raw.copy().pick_channels(channels)
    raw_filtered_patch = raw_filtered.copy().pick_channels(channels)

    # Drop the channels we've select to run from the overall raw
    raw = raw.drop_channels(channels)

    ica = mne.preprocessing.ICA(n_components=len(raw_filtered_patch.ch_names), max_iter='auto',
                                random_state=97)
    ica.fit(raw_filtered_patch)

    raw_patch.load_data()

    # Automatically choose ICA components
    ica.exclude = []
    # find which ICs match the ECG pattern
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw_patch, ch_name='ECG')
    ica.exclude = ecg_indices

    # Just for visualising
    # ica.plot_overlay(raw_patch.copy().drop_channels(['ECG']), exclude=ecg_indices, picks='eeg')
    # print(ica.exclude)
    # ica.plot_scores(ecg_scores)

    # Apply the ica we got from the filtered data onto the unfiltered raw
    ica.apply(raw_patch)

    raw.add_channels([raw_patch])  # Add back the channels I removed

    # Filter
    raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
               iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

    raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

    # Save data
    fname = 'separated_clean_baseline_ica_auto_' + cond_name + '.fif'
    raw.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

