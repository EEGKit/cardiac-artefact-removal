# Want to get negative amplitude in relevant time period for reliability assessment

import mne
import numpy as np
import h5py
from scipy.io import loadmat
from SNR_functions import *
from epoch_data import rereference_data
import matplotlib.pyplot as plt


if __name__ == '__main__':

    subjects = np.arange(1, 37)  # (1, 37) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    # Loop through methods and save as required
    which_method = {'Prep': True,
                    'PCA': True,
                    'ICA': True,
                    'SSP_5': True,
                    'SSP_6': True}

    for i in np.arange(0, len(which_method)):
        method = list(which_method.keys())[i]
        if which_method[method]:  # If this method is true, go through with the rest
            class save_Amp():
                def __init__(self):
                    pass

            # Instantiate class
            saveamp = save_Amp()

            # Matrix of dimensions no.subjects x no. trials
            amp_med = np.zeros((len(subjects), 2000))
            amp_tib = np.zeros((len(subjects), 2000))

            for subject in subjects:
                for cond_name in cond_names:
                    if cond_name == 'tibial':
                        trigger_name = 'Tibial - Stimulation'
                        nerve = 2
                        channels = ['S23', 'L1', 'S31']
                        start = 12 / 1000
                        end = 32 / 1000

                    elif cond_name == 'median':
                        trigger_name = 'Median - Stimulation'
                        nerve = 1
                        channels = ['S6', 'SC6', 'S14']
                        start = 8 / 1000
                        end = 18 / 1000

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Get the right file path
                    if method == 'Prep':
                        file_path = "/data/pt_02569/tmp_data/prepared_py/"
                        file_name = f'noStimart_sr1000_{cond_name}_withqrs.fif'
                    elif method == 'PCA':
                        file_path = "/data/pt_02569/tmp_data/ecg_rm_py/"
                        file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                    elif method == 'ICA':
                        file_path = "/data/pt_02569/tmp_data/baseline_ica_py/"
                        file_name = f'clean_baseline_ica_auto_{cond_name}.fif'
                    elif method == 'SSP_5' or method == 'SSP_6':
                        file_path = "/data/p_02569/SSP/"
                        file_name = f'ssp_cleaned_{cond_name}.fif'

                    if method == 'SSP_5':
                        input_path = file_path + subject_id + '/5 projections/'
                    elif method == 'SSP_6':
                        input_path = file_path + subject_id + '/6 projections/'
                    else:
                        input_path = file_path + subject_id + "/esg/prepro/"

                    raw = mne.io.read_raw_fif(f"{input_path}{file_name}", preload=True)

                    if method == 'Prep' or method == 'PCA':
                        # add reference channel to data
                        mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                        raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                                   iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                        raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                    # Form epochs
                    events, event_ids = mne.events_from_annotations(raw)
                    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                        baseline=tuple(iv_baseline))
                    no_epochs = np.shape(epochs.get_data())[0]

                    # Get minimum amplitude of each epoch in expected time period
                    for trial in np.arange(0, no_epochs):
                        single_epoch = epochs[trial]
                        evoked = single_epoch.average()  # Just to kill a dimension

                        mins = []
                        for ch in channels:
                            # Pick one channel at a time, get minimum in time range
                            evoked_channel = evoked.copy().pick_channels([ch])
                            time_idx = evoked_channel.time_as_index([start, end])
                            data = evoked_channel.data[0, time_idx[0]:time_idx[1]]  # only one channel, time period
                            mins.append(np.min(data))

                        # Get absolute minimum out of the three possible channels
                        min = np.min(mins)

                        # Now have one amp related to each subject and trial
                        if cond_name == 'median':
                            amp_med[subject - 1, trial] = min
                        elif cond_name == 'tibial':
                            amp_tib[subject - 1, trial] = min

            # Save to file
            saveamp.amp_med = amp_med
            saveamp.amp_tib = amp_tib
            dataset_keywords = [a for a in dir(saveamp) if not a.startswith('__')]
            if method == 'SSP_5':
                fn = f"{file_path}amplitudes_5.h5"
            elif method == 'SSP_6':
                fn = f"{file_path}amplitudes_6.h5"
            else:
                fn = f"{file_path}amplitudes.h5"
            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(saveamp, keyword))
