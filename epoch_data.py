import mne
from get_conditioninfo import *
from scipy.io import loadmat
import os

def rereference_data(raw, ch_name):
    if ch_name in raw.ch_names:
        raw_ref = raw.copy().set_eeg_reference(ref_channels=[ch_name])
    else:
        raw_ref = raw.copy().set_eeg_reference(ref_channels='average')

    return raw_ref


def epoch_data(subject, condition, srmr_nr, sampling_rate):
    # set variables
    subject_id = f'sub-{str(subject).zfill(3)}'
    load_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"  # Taking data from the ecg_rm_py folder
    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    save_path = "/data/pt_02569/tmp_data/epoched_py/" + subject_id + "/esg/prepro/"
    os.makedirs(save_path, exist_ok=True)

    # get condition info
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    nerve = cond_info.nerve
    nblocks = cond_info.nblocks

    # load cleaned ESG data
    fname = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
    raw = mne.io.read_raw_fif(load_path + fname, preload=True)

    ########################### Re - Reference ####################################
    # add reference channel to data
    mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

    # Fz reference
    raw_FzRef = rereference_data(raw, 'Fz-TH6')

    # anterior reference
    if nerve == 1:
        raw_antRef = rereference_data(raw, 'AC')
    elif nerve == 2:
        raw_antRef = rereference_data(raw, 'AL')

    ########################### Filtering ############################################
    cfg = loadmat(cfg_path+'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
               iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    raw_FzRef.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw_FzRef.ch_names), method='iir',
                     iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    raw_antRef.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw_antRef.ch_names), method='iir',
                      iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

    raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    raw_FzRef.notch_filter(freqs=notch_freq, n_jobs=len(raw_FzRef.ch_names), method='fir', phase='zero')
    raw_antRef.notch_filter(freqs=notch_freq, n_jobs=len(raw_antRef.ch_names), method='fir', phase='zero')

    ############################################# Epoch ##############################################
    # Both in ms - MNE works with seconds
    iv_epoch = cfg['iv_epoch'][0]/1000
    iv_baseline = cfg['iv_baseline'][0]/1000

    # events contains timestamps with corresponding event_id (number)
    # event_ids returns the event/trigger names with their corresponding event_id (number)
    events, event_ids = mne.events_from_annotations(raw)
    events_antRef, event_ids_antRef = mne.events_from_annotations(raw_antRef)
    events_FzRef, event_ids_FzRef = mne.events_from_annotations(raw_FzRef)


    # Extract our event of interest as a dictionary so the keys can later be used to access associated events.
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    event_id_dict_ant = {key: value for key, value in event_ids_antRef.items() if key == trigger_name}
    event_id_dict_Fz = {key: value for key, value in event_ids_FzRef.items() if key == trigger_name}

    # Cut epochs
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                        baseline=tuple(iv_baseline))
    epochs_antRef = mne.Epochs(raw_antRef, events_antRef, event_id=event_id_dict_ant, tmin=iv_epoch[0], tmax=iv_epoch[1]
                               , baseline=tuple(iv_baseline))
    epochs_FzRef = mne.Epochs(raw_FzRef, events_FzRef, event_id=event_id_dict_Fz, tmin=iv_epoch[0], tmax=iv_epoch[1],
                              baseline=tuple(iv_baseline))

    # Save the epochs
    fname = 'epo_clean_' + cond_name + '.fif'
    epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)
    fname = 'epo_antRef_clean_' + cond_name + '.fif'
    epochs_antRef.save(os.path.join(save_path, fname), fmt='double', overwrite=True)
    fname = 'epo_FzRef_clean_' + cond_name + '.fif'
    epochs_FzRef.save(os.path.join(save_path, fname), fmt='double', overwrite=True)



