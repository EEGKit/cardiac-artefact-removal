# File to compuute a different type of SNR for the heart artefact reduction
# 1. Get the RMS of the heart artefact area - (-200ms, 400ms) about the R-peak
# 2. Get the RMS of a baseline period - (-350ms, -250ms) in relation to the R-peak
# Divide 1 by 2 for a measure of the heart artefact SNR
# To see if taking just the R-peak is too specific to capture heart artefact reduction

import mne
import numpy as np
import h5py
from scipy.io import loadmat


def compute_rms_snr(Evoked, ivb, ivh, channels):
    iv_baseline = Evoked.time_as_index(ivb)
    iv_heart = Evoked.time_as_index(ivh)
    # evoked.data = data array of shape(n_channels, n_times)
    rms = []
    for ch in np.arange(0, len(channels)):
        # Pick a single channel
        data_h = Evoked.data[ch, iv_heart[0]:iv_heart[1]]
        data_b = Evoked.data[ch, iv_baseline[0]:iv_baseline[1]]
        rms_h = np.sqrt(np.mean(data_h ** 2))
        rms_b = np.sqrt(np.mean(data_b ** 2))
        rms.append(rms_h / rms_b)

    return rms


if __name__ == '__main__':
    # Set which to run
    calc_raw = False
    calc_PCA = False
    calc_ICA = False
    calc_post_ICA = False
    calc_SSP = False

    # Testing with just subject 1 at the moment
    subjects = np.arange(1, 37)  # (1, 2) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    iv_b = [-350/1000, -250/1000]      # Baseline period we want RMS of
    iv_h = [-200 / 1000, 400 / 1000]  # Heartbeat area we want RMS of
    iv_epoch = [-350/1000, 400/1000]  # Entire epoch to extract

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    #########################################
    # Raw SNR Calculations
    #########################################
    if calc_raw:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 39))
        snr_tib = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from preparation script
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                fname = f"noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # add reference channel to data
                mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                # Extract relevant epochs
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == 'qrs'}
                epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1])

                evoked = epochs.average()

                # Now we have an evoked potential about the heartbeat
                # Want to compute the RMS for heartbeat versus baseline
                rms = compute_rms_snr(evoked, iv_b, iv_h, esg_chans)

                # Now have rms snr related to each subject (39 channels) and condition
                if cond_name == 'median':
                    snr_med[subject - 1, :] = rms
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, :] = rms

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/prepared_py/rms_snr_heart.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))


    #########################################
    # PCA SNR Calculations
    #########################################
    if calc_PCA:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 39))
        snr_tib = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from PCA script
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # add reference channel to data
                mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                # Extract relevant epochs
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == 'qrs'}
                epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1])

                evoked = epochs.average()

                # Now we have an evoked potential about the heartbeat
                # Want to compute the RMS for heartbeat versus baseline
                rms = compute_rms_snr(evoked, iv_b, iv_h, esg_chans)

                # Now have rms snr related to each subject (39 channels) and condition
                if cond_name == 'median':
                    snr_med[subject - 1, :] = rms
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, :] = rms

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/ecg_rm_py/rms_snr_heart.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))


    #########################################
    # ICA SNR Calculations
    #########################################
    if calc_ICA:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 39))
        snr_tib = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load epochs resulting from ICA
                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Extract relevant epochs
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == 'qrs'}
                epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1])

                evoked = epochs.average()

                # Now we have an evoked potential about the heartbeat
                # Want to compute the RMS for heartbeat versus baseline
                rms = compute_rms_snr(evoked, iv_b, iv_h, esg_chans)

                # Now have rms snr related to each subject (39 channels) and condition
                if cond_name == 'median':
                    snr_med[subject - 1, :] = rms
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, :] = rms

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/baseline_ica_py/rms_snr_heart.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    #########################################
    # Post ICA SNR Calculations
    #########################################
    if calc_post_ICA:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 39))
        snr_tib = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load epochs resulting from Post-ICA
                input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
                fname = f"clean_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Extract relevant epochs
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == 'qrs'}
                epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1])

                evoked = epochs.average()

                # Now we have an evoked potential about the heartbeat
                # Want to compute the RMS for heartbeat versus baseline
                rms = compute_rms_snr(evoked, iv_b, iv_h, esg_chans)

                # Now have rms snr related to each subject (39 channels) and condition
                if cond_name == 'median':
                    snr_med[subject - 1, :] = rms
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, :] = rms

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/ica_py/rms_snr_heart.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    #########################################
    # SSP SNR Calculations
    #########################################
    if calc_SSP:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        for n in np.arange(5, 21):
            # Matrix of dimensions no.subjects x no. projections
            snr_med = np.zeros((len(subjects), 39))
            snr_tib = np.zeros((len(subjects), 39))

            for subject in subjects:
                for cond_name in cond_names:

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Want the SNR
                    # Load epochs resulting from SSP
                    input_path = "/data/p_02569/SSP/" + subject_id
                    savename = input_path + "/" + str(n) + " projections/"
                    raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")

                    # Extract relevant epochs
                    events, event_ids = mne.events_from_annotations(raw)
                    event_id_dict = {key: value for key, value in event_ids.items() if key == 'qrs'}
                    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1])

                    evoked = epochs.average()

                    # Now we have an evoked potential about the heartbeat
                    # Want to compute the RMS for heartbeat versus baseline
                    rms = compute_rms_snr(evoked, iv_b, iv_h, esg_chans)

                    # Now have rms snr related to each subject (39 channels) and condition
                    if cond_name == 'median':
                        snr_med[subject - 1, :] = rms
                    elif cond_name == 'tibial':
                        snr_tib[subject - 1, :] = rms

            # Save to file to compare to matlab - only for debugging
            savesnr.snr_med = snr_med
            savesnr.snr_tib = snr_tib
            dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

            fn = f"/data/p_02569/SSP/rms_snr_heart_{n}.h5"

            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    #############################################################
    # Now we have them saved, print to screen
    #############################################################
    # All being read in have have n_subjects x n_channels (36, 39)
    keywords = ['snr_med', 'snr_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/rms_snr_heart.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        snr_med_prep = infile[keywords[0]][()]
        snr_tib_prep = infile[keywords[1]][()]

    print(f"Prep Medial: {np.mean(snr_med_prep, axis=tuple([0, 1])):.4}")
    print(f"Prep Tibial: {np.mean(snr_tib_prep, axis=tuple([0, 1])):.4}")

    # PCA
    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/rms_snr_heart.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        snr_med_pca = infile[keywords[0]][()]
        snr_tib_pca = infile[keywords[1]][()]

    print(f"PCA Medial: {np.mean(snr_med_pca, axis=tuple([0, 1])):.4}")
    print(f"PCA Tibial: {np.mean(snr_tib_pca, axis=tuple([0, 1])):.4}")

    # ICA
    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/rms_snr_heart.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        snr_med_ica = infile[keywords[0]][()]
        snr_tib_ica = infile[keywords[1]][()]

    print(f"ICA Medial: {np.mean(snr_med_ica, axis=tuple([0, 1])):.4}")
    print(f"ICA Tibial: {np.mean(snr_tib_ica, axis=tuple([0, 1])):.4}")

    # Post-ICA
    fn = f"/data/pt_02569/tmp_data/ica_py/rms_snr_heart.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        snr_med_post_ica = infile[keywords[0]][()]
        snr_tib_post_ica = infile[keywords[1]][()]

    print(f"Post-ICA Medial: {np.mean(snr_med_post_ica, axis=tuple([0, 1])):.4}")
    print(f"Post-ICA Tibial: {np.mean(snr_tib_post_ica, axis=tuple([0, 1])):.4}")

    # SSP
    for n in np.arange(5, 21):
        fn = f"/data/p_02569/SSP/rms_snr_heart_{n}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            snr_med_ssp = infile[keywords[0]][()]
            snr_tib_ssp = infile[keywords[1]][()]

        print(f"SSP Medial {n}: {np.mean(snr_med_ssp, axis=tuple([0, 1])):.4}")
        print(f"SSP Tibial {n}: {np.mean(snr_tib_ssp, axis=tuple([0, 1])):.4}")


    ###################################################
    # Channels of Interest
    ###################################################
    print('\n')
    median_pos = []
    tibial_pos = []
    for channel in ['S23', 'L1', 'S31']:
        tibial_pos.append(esg_chans.index(channel))
    for channel in ['S6', 'SC6', 'S14']:
        median_pos.append(esg_chans.index(channel))

    fn = f"/data/pt_02569/tmp_data/prepared_py/rms_snr_heart.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        snr_med_prep = infile[keywords[0]][()]
        snr_tib_prep = infile[keywords[1]][()]

    print(f"Prep Medial: {np.mean(snr_med_prep[:, median_pos], axis=tuple([0, 1])):.4}")
    print(f"Prep Tibial: {np.mean(snr_tib_prep[:, tibial_pos], axis=tuple([0, 1])):.4}")

    # PCA
    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/rms_snr_heart.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        snr_med_pca = infile[keywords[0]][()]
        snr_tib_pca = infile[keywords[1]][()]

    print(f"PCA Medial: {np.mean(snr_med_pca[:, median_pos], axis=tuple([0, 1])):.4}")
    print(f"PCA Tibial: {np.mean(snr_tib_pca[:, tibial_pos], axis=tuple([0, 1])):.4}")

    # ICA
    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/rms_snr_heart.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        snr_med_ica = infile[keywords[0]][()]
        snr_tib_ica = infile[keywords[1]][()]

    print(f"ICA Medial: {np.mean(snr_med_ica[:, median_pos], axis=tuple([0, 1])):.4}")
    print(f"ICA Tibial: {np.mean(snr_tib_ica[:, tibial_pos], axis=tuple([0, 1])):.4}")

    # Post-ICA
    fn = f"/data/pt_02569/tmp_data/ica_py/rms_snr_heart.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        snr_med_post_ica = infile[keywords[0]][()]
        snr_tib_post_ica = infile[keywords[1]][()]

    print(f"Post-ICA Medial: {np.mean(snr_med_post_ica[:, median_pos], axis=tuple([0, 1])):.4}")
    print(f"Post-ICA Tibial: {np.mean(snr_tib_post_ica[:, tibial_pos], axis=tuple([0, 1])):.4}")

    # SSP
    for n in np.arange(5, 21):
        fn = f"/data/p_02569/SSP/rms_snr_heart_{n}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            snr_med_ssp = infile[keywords[0]][()]
            snr_tib_ssp = infile[keywords[1]][()]

        print(f"SSP Medial {n}: {np.mean(snr_med_ssp[:, median_pos], axis=tuple([0, 1])):.4}")
        print(f"SSP Tibial {n}: {np.mean(snr_tib_ssp[:, tibial_pos], axis=tuple([0, 1])):.4}")

