# Script to compute the improved normalised power spectrum
# Formula is the sum of the power in the first n harmonics of the contaminated signal, divided by the same of the
# reconstructed signal
# The INPS ratio is widely used in existing studies for assessing the residuals of the BCG artefact in
# reconstructed EEG data [50]. In our study, the fundamental frequency of the BCG (calculated using the
# ECG signal, which is approximately 1 Hz) and its four harmonics have been used for computing the INPS
# Javed paper 2017

# The  lower  the  value  of  the  INPS  ratio,  the  more  the  residuals  of  the  BCG  are present in
# the reconstructed dataset
# WANT A HIGHER INPS RATIO

# Need to compute the heartrate in bpm, convert then to Hz
# Compute the first 4 harmonics of this frequency
# Compute the power at the fundamental freq and first 4 harmonics and sum them
# Will first try compute power with bandpower function, then will try goertzel.py (more complex)
# When we have all these saved, as with residual_intensity.py we can compute the ratios

from scipy.io import loadmat
from goertzel import goertzel
import scipy
import numpy as np
import mne
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import matplotlib.pyplot as plt
import h5py
from SNR_functions import evoked_from_raw


def get_harmonics(raw1, trigger, sample_rate):
    minutes = (raw1.times[-1])/60  # Gets end time in seconds, convert to minutes
    events, event_ids = mne.events_from_annotations(raw1)
    relevant_events = mne.pick_events(events, include=event_ids[trigger])
    heart_rate = max(np.shape(relevant_events))/minutes  # This gives the number of R-peaks divided by time in min
    freqs = []
    f_0 = heart_rate/60
    freqs.append(f_0)
    for nq in np.arange(2, 6):
        freqs.append(nq*f_0)

    return freqs


def power_at_harmonics(raw_data, channels, fs, harmonics):
    # Pick channels
    rel_data = raw_data.get_data(picks=channels)
    N_samp = raw_data.n_times

    # Output an array of shape n_channels x n_frequencies
    y_f = rfft(rel_data, workers=len(channels))  # This allows the computation of all channels concurrently
    # Calculates the frequencies in the center of each bin in the output of rfft()
    x_f = rfftfreq(N_samp, 1 / fs)

    # Get power spectrum from fft
    abs_ft = np.abs(y_f)
    p_spec = np.square(abs_ft)

    p = []
    for i in np.arange(0, 5):
        first = np.where(x_f > harmonics[i] - 0.005)
        last = np.where(x_f < harmonics[i] + 0.005)
        indices = np.intersect1d(first, last)
        # power.append(np.sum(p_spec[:, indices]))  # Gets the sum across all channels at those indices
        # Gets the sum over the indices of interest in this harmonic
        p.append(np.sum(p_spec[:, indices], axis=1))

    # Gets the total power over all harmonics for this subject
    # Still have one value per channel
    p = np.sum(p, axis=0)

    return p


if __name__ == '__main__':

    calc_prepared = True  # Should always be true as this is the baseline we get the ratio with
    calc_PCA = True
    calc_post_ICA = True
    calc_ICA = True
    calc_SSP = True
    reduced_epochs = False  # Dummy variable - always false in this script as I don't reduce epochs

    # Define the channel names so they come out of each dataset the same
    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    # Testing with just subject 1 at the moment
    subjects = np.arange(1, 37)  # (1, 37) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')

    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    ##########################################################################
    # Calculate Power for Prepared Data
    ##########################################################################
    if calc_prepared:
        class save_pow():
            def __init__(self):
                pass

        # Instantiate class
        savepow = save_pow()

        # Matrix of dimensions no.subjects x no_channels
        pow_med_prep = np.zeros((len(subjects), 39))
        pow_tib_prep = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'qrs'
                    nerve = 2
                elif cond_name == 'median':
                    trigger_name = 'qrs'
                    nerve = 1

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the RMS of the data
                # Load epochs resulting from importing the data - the raw data in this folder has not been rereferenced
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif",
                                          preload=True)

                frequencies = get_harmonics(raw, trigger_name, sampling_rate)

                mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                # Now we have the raw data in the filtered form
                # We have the fundamental frequency and harmonics for this subject
                # Compute power at the frequencies
                power = power_at_harmonics(raw, esg_chans, sampling_rate, frequencies)

                # Now have power for each subject, insert it into the correct condition array
                if cond_name == 'median':
                    pow_med_prep[subject - 1, :] = power
                elif cond_name == 'tibial':
                    pow_tib_prep[subject - 1, :] = power

        # Save to file
        savepow.pow_med = pow_med_prep
        savepow.pow_tib = pow_tib_prep
        dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/prepared_py/inps.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savepow, keyword))

    ##########################################################################
    # Calculate Power for PCA Data
    ##########################################################################
    if calc_PCA:
        class save_pow():
            def __init__(self):
                pass

        # Instantiate class
        savepow = save_pow()

        # Matrix of dimensions no.subjects x no. channels
        pow_med_pca = np.zeros((len(subjects), 39))
        pow_tib_pca = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'qrs'
                    nerve = 2
                elif cond_name == 'median':
                    trigger_name = 'qrs'
                    nerve = 1

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the RMS of the data
                # Load epochs resulting from PCA - the raw data in this folder has not been rereferenced
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path+fname, preload=True)

                frequencies = get_harmonics(raw, trigger_name, sampling_rate)

                mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                # Now we have the raw data in the filtered form
                # We have the fundamental frequency and harmonics for this subject
                # Compute power at the frequencies
                power = power_at_harmonics(raw, esg_chans, sampling_rate, frequencies)

                # Now have power for each subject, insert it into the correct condition array
                if cond_name == 'median':
                    pow_med_pca[subject - 1, :] = power
                elif cond_name == 'tibial':
                    pow_tib_pca[subject - 1, :] = power

        # Save to file
        savepow.pow_med = pow_med_pca
        savepow.pow_tib = pow_tib_pca
        dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/ecg_rm_py/inps.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savepow, keyword))

    ##########################################################################
    # Calculate Power for ICA Data
    ##########################################################################
    if calc_ICA:
        class save_pow():
            def __init__(self):
                pass

        # Instantiate class
        savepow = save_pow()

        # Matrix of dimensions no.subjects x no. channels
        pow_med_ica = np.zeros((len(subjects), 39))
        pow_tib_ica = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'qrs'
                    nerve = 2
                elif cond_name == 'median':
                    trigger_name = 'qrs'
                    nerve = 1

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the RMS of the data
                # Load epochs resulting from ICA
                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                frequencies = get_harmonics(raw, trigger_name, sampling_rate)

                # Now we have the raw data in the filtered form
                # We have the fundamental frequency and harmonics for this subject
                # Compute power at the frequencies
                power = power_at_harmonics(raw, esg_chans, sampling_rate, frequencies)

                # Now have power for each subject, insert it into the correct condition array
                if cond_name == 'median':
                    pow_med_ica[subject - 1, :] = power
                elif cond_name == 'tibial':
                    pow_tib_ica[subject - 1, :] = power

        # Save to file
        savepow.pow_med = pow_med_ica
        savepow.pow_tib = pow_tib_ica
        dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savepow, keyword))


    ##########################################################################
    # Calculate Power for Post ICA Data
    ##########################################################################
    if calc_post_ICA:
        class save_pow():
            def __init__(self):
                pass

        # Instantiate class
        savepow = save_pow()

        # Matrix of dimensions no.subjects x no. channels
        pow_med_post_ica = np.zeros((len(subjects), 39))
        pow_tib_post_ica = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'qrs'
                    nerve = 2
                elif cond_name == 'median':
                    trigger_name = 'qrs'
                    nerve = 1

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the RMS of the data
                # Load epochs resulting from ICA the data - the raw data in this folder has not been rereferenced
                input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
                fname = f"clean_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                frequencies = get_harmonics(raw, trigger_name, sampling_rate)

                # Now we have the raw data in the filtered form
                # We have the fundamental frequency and harmonics for this subject
                # Compute power at the frequencies
                power = power_at_harmonics(raw, esg_chans, sampling_rate, frequencies)

                # Now have power for each subject, insert it into the correct condition array
                if cond_name == 'median':
                    pow_med_post_ica[subject - 1, :] = power
                elif cond_name == 'tibial':
                    pow_tib_post_ica[subject - 1, :] = power

        # Save to file
        savepow.pow_med = pow_med_post_ica
        savepow.pow_tib = pow_tib_post_ica
        dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/ica_py/inps.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savepow, keyword))

    ##########################################################################
    # Calculate Power for SSP Data
    ##########################################################################
    if calc_SSP:
        class save_pow():
            def __init__(self):
                pass

        # Instantiate class
        savepow = save_pow()

        for n in np.arange(5, 21):
            # Matrix of dimensions no.subjects x no. channel
            pow_med_ssp = np.zeros((len(subjects), 39))
            pow_tib_ssp = np.zeros((len(subjects), 39))

            for subject in subjects:
                for cond_name in cond_names:
                    if cond_name == 'tibial':
                        trigger_name = 'qrs'
                        nerve = 2
                    elif cond_name == 'median':
                        trigger_name = 'qrs'
                        nerve = 1

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Want the RMS of the data
                    # Load epochs resulting from SSP the data
                    input_path = "/data/p_02569/SSP/" + subject_id
                    savename = input_path + "/" + str(n) + " projections/"
                    raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")

                    frequencies = get_harmonics(raw, trigger_name, sampling_rate)

                    # Now we have the raw data in the filtered form
                    # We have the fundamental frequency and harmonics for this subject
                    # Compute power at the frequencies
                    power = power_at_harmonics(raw, esg_chans, sampling_rate, frequencies)

                    # Now have power for each subject, insert it into the correct condition array
                    if cond_name == 'median':
                        pow_med_ssp[subject - 1, :] = power
                    elif cond_name == 'tibial':
                        pow_tib_ssp[subject - 1, :] = power

            # Save to file
            savepow.pow_med = pow_med_ssp
            savepow.pow_tib = pow_tib_ssp
            dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]

            fn = f"/data/p_02569/SSP/inps_{n}.h5"

            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savepow, keyword))


    ##########################################################################
    # Calculate INPS for each - Prepared divided by clean
    ##########################################################################
