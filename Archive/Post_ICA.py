# Apply auto ICA on the data already cleaned using PCA_OBS

import os
import mne
import matplotlib.pyplot as plt
from scipy.io import loadmat
from get_conditioninfo import *
from epoch_data import rereference_data


def run_post_ica(subject, condition, srmr_nr, sampling_rate):
    # Set paths
    subject_id = f'sub-{str(subject).zfill(3)}'
    save_path = "../tmp_data/ica_py/" + subject_id + "/esg/prepro/"  # Saving to ica_py
    input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"  # Taking data from PCA_OBS output
    figure_path = "/data/p_02569/ICA_images/" + subject_id + "/"
    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    # Get the condition information based on the condition read in
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    nerve = cond_info.nerve

    # load cleaned ESG data
    fname = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # make a copy to filter
    raw_filtered = raw.copy().drop_channels(['ECG'])

    # filtering
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    raw_filtered.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

    raw_filtered.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

    # ICA
    ica = mne.preprocessing.ICA(n_components=len(raw_filtered.ch_names), max_iter='auto', random_state=97)
    # ica = mne.preprocessing.ICA(n_components=len(raw_filtered.ch_names), max_iter='auto', random_state=97)
    ica.fit(raw_filtered)

    raw.load_data()
    # ica.plot_sources(raw, show_scrollbars=False)
    # ica.plot_overlay(raw, exclude=[0], picks='eeg')
    # ica.plot_overlay(raw, exclude=[0, 1], picks='eeg')
    # ica.plot_overlay(raw, exclude=[0, 1, 2], picks='eeg')
    # ica.plot_overlay(raw, exclude=[0, 1, 2, 3], picks='eeg')
    # plt.show()

    # Indices chosen based on plot above
    # ica.exclude = [0, 1, 2]

    # Automatically choose ICA components
    ica.exclude = []
    # find which ICs match the EOG pattern
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG')
    ica.exclude = ecg_indices
    # ica.plot_scores(ecg_scores)

    # Apply the ica we got from the filtered data onto the unfiltered raw
    ica.apply(raw)

    # Fz reference
    raw_FzRef = rereference_data(raw, 'Fz-TH6')

    # anterior reference
    if nerve == 1:
        raw_antRef = rereference_data(raw, 'AC')
    elif nerve == 2:
        raw_antRef = rereference_data(raw, 'AL')

    # add reference channel to data - average rereferencing
    mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

    raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
               iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    raw_FzRef.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw_FzRef.ch_names), method='iir',
                     iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    raw_antRef.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw_antRef.ch_names), method='iir',
                      iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

    raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    raw_FzRef.notch_filter(freqs=notch_freq, n_jobs=len(raw_FzRef.ch_names), method='fir', phase='zero')
    raw_antRef.notch_filter(freqs=notch_freq, n_jobs=len(raw_antRef.ch_names), method='fir', phase='zero')

    # Save raw data - not epoched
    fname = 'clean_ica_auto_' + cond_name + '.fif'
    raw.save(os.path.join(save_path, fname), fmt='double', overwrite=True)
    fname = 'clean_ica_auto_antRef_' + cond_name + '.fif'
    raw_antRef.save(os.path.join(save_path, fname), fmt='double', overwrite=True)
    fname = 'clean_ica_auto_FzRef_' + cond_name + '.fif'
    raw_FzRef.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    # Perform epoching
    # Both in ms - MNE works with seconds
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    # events contains timestamps with corresponding event_id (number)
    # event_ids returns the event/trigger names with their corresponding event_id (number)
    events, event_ids = mne.events_from_annotations(raw)

    # Extract our event of interest as a dictionary so the keys can later be used to access associated events.
    # trigger_name is obtained at the top of the script from get_conditioninfo script
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}

    # Cut epochs, can remove baseline within this step also by specifying baseline period
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                        baseline=tuple(iv_baseline))

    # Save the epochs
    fname = 'epo_clean_ica_auto_' + cond_name + '.fif'
    epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    # Plot certain channels for visual inspection afterwards
    channelNames = ['S35', 'Iz', 'SC1', 'S3', 'SC6', 'S20', 'L1', 'L4']

    # Can use combine='mean' to get mean across channels selected
    for count in np.arange(0, len(channelNames)):
        # Try extracting the evoked signal and just plotting that with dashed line
        fig = epochs[trigger_name].plot_image(picks=channelNames[count], vmin=-5, vmax=5, show=False)
        plt.savefig(f"{figure_path}{channelNames[count]}_{cond_name}_{len(ica.exclude)}_ICs_auto.jpg")
        plt.close()
