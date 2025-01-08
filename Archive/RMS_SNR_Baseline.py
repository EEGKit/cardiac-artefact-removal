# With this script we want to look at other effects SSP is having on noise
# Extract an epoch about the heart artefact - -350 to 400ms
# Calculate the RMS in the baseline period (-350ms, -250ms) for raw data + each cleaning method
# Calculate the residuals in the baseline period - RMS clean / RMS dirty

import mne
import numpy as np
import h5py
from scipy.io import loadmat


def compute_rms(Evoked, iv, channels):
    Evoked.reorder_channels(channels)  # Force order of channels
    iv_baseline = Evoked.time_as_index(iv)
    # evoked.data = data array of shape(n_channels, n_times)
    rms = []
    for ch in np.arange(0, len(channels)):
        # Pick a single channel
        data_b = Evoked.data[ch, iv_baseline[0]:iv_baseline[1]]
        rms_b = np.sqrt(np.mean(data_b ** 2))
        rms.append(rms_b)

    return rms


if __name__ == '__main__':
    # Set which to run
    calc_raw = False
    calc_PCA = False
    calc_ICA = False
    calc_SSP = True

    # Testing with just subject 1 at the moment
    subjects = np.arange(1, 37)  # (1, 2) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    iv_b = [-350/1000, -250/1000]      # Baseline period we want RMS of
    iv_epoch = [-350/1000, 400/1000]  # Entire epoch to extract relative to R-peak

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
    # Raw RMS Calculations
    #########################################
    if calc_raw:
        # Declare class to hold ecg fit information
        class save_RMS():
            def __init__(self):
                pass

        # Instantiate class
        saverms = save_RMS()

        # Matrix of dimensions no.subjects x no. projections
        rms_med = np.zeros((len(subjects), 39))
        rms_tib = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from preparation script
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
                fname = f"noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Extract relevant epochs
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == 'qrs'}
                epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1])

                evoked = epochs.average()

                # Now we have an evoked potential about the heartbeat
                # Want to compute the RMS for baseline period
                rms = compute_rms(evoked, iv_b, esg_chans)

                # Now have rms snr related to each subject (39 channels) and condition
                if cond_name == 'median':
                    rms_med[subject - 1, :] = rms
                elif cond_name == 'tibial':
                    rms_tib[subject - 1, :] = rms

        # Save to file to compare to matlab - only for debugging
        saverms.snr_med = rms_med
        saverms.snr_tib = rms_tib
        dataset_keywords = [a for a in dir(saverms) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/prepared_py/rms_baseline.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(saverms, keyword))

    #########################################
    # PCA RMS Calculations
    #########################################
    if calc_PCA:
        # Declare class to hold ecg fit information
        class save_RMS():
            def __init__(self):
                pass

        # Instantiate class
        saverms = save_RMS()

        # Matrix of dimensions no.subjects x no. projections
        rms_med = np.zeros((len(subjects), 39))
        rms_tib = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from PCA script
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Extract relevant epochs
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == 'qrs'}
                epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1])

                evoked = epochs.average()

                # Now we have an evoked potential about the heartbeat
                # Want to compute the RMS for baseline period
                rms = compute_rms(evoked, iv_b, esg_chans)

                # Now have rms snr related to each subject (39 channels) and condition
                if cond_name == 'median':
                    rms_med[subject - 1, :] = rms
                elif cond_name == 'tibial':
                    rms_tib[subject - 1, :] = rms

        # Save to file to compare to matlab - only for debugging
        saverms.snr_med = rms_med
        saverms.snr_tib = rms_tib
        dataset_keywords = [a for a in dir(saverms) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/ecg_rm_py/rms_baseline.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(saverms, keyword))

    #########################################
    # ICA RMS Calculations
    #########################################
    if calc_ICA:
        # Declare class to hold ecg fit information
        class save_RMS():
            def __init__(self):
                pass

        # Instantiate class
        saverms = save_RMS()

        # Matrix of dimensions no.subjects x no. projections
        rms_med = np.zeros((len(subjects), 39))
        rms_tib = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from ICA script
                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id
                fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Extract relevant epochs
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == 'qrs'}
                epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1])

                evoked = epochs.average()

                # Now we have an evoked potential about the heartbeat
                # Want to compute the RMS for baseline period
                rms = compute_rms(evoked, iv_b, esg_chans)

                # Now have rms snr related to each subject (39 channels) and condition
                if cond_name == 'median':
                    rms_med[subject - 1, :] = rms
                elif cond_name == 'tibial':
                    rms_tib[subject - 1, :] = rms

        # Save to file to compare to matlab - only for debugging
        saverms.snr_med = rms_med
        saverms.snr_tib = rms_tib
        dataset_keywords = [a for a in dir(saverms) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/baseline_ica_py/rms_baseline.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(saverms, keyword))

    #########################################
    # SSP RMS Calculations
    #########################################
    if calc_SSP:
        # Declare class to hold ecg fit information
        class save_RMS():
            def __init__(self):
                pass

        # Instantiate class
        saverms = save_RMS()

        for n in np.arange(5, 21):
            # Matrix of dimensions no.subjects x no. projections
            rms_med = np.zeros((len(subjects), 39))
            rms_tib = np.zeros((len(subjects), 39))

            for subject in subjects:
                for cond_name in cond_names:

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Want the SNR
                    # Load data resulting from SSP script
                    input_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id
                    savename = input_path + "/" + str(n) + " projections/"
                    raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")

                    # Extract relevant epochs
                    events, event_ids = mne.events_from_annotations(raw)
                    event_id_dict = {key: value for key, value in event_ids.items() if key == 'qrs'}
                    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1])

                    evoked = epochs.average()

                    # Now we have an evoked potential about the heartbeat
                    # Want to compute the RMS for baseline period
                    rms = compute_rms(evoked, iv_b, esg_chans)

                    # Now have rms snr related to each subject (39 channels) and condition
                    if cond_name == 'median':
                        rms_med[subject - 1, :] = rms
                    elif cond_name == 'tibial':
                        rms_tib[subject - 1, :] = rms

            # Save to file to compare to matlab - only for debugging
            saverms.snr_med = rms_med
            saverms.snr_tib = rms_tib
            dataset_keywords = [a for a in dir(saverms) if not a.startswith('__')]

            fn = f"/data/pt_02569/tmp_data/ssp_py/rms_baseline_{n}.h5"

            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(saverms, keyword))

    ################################################################################
    # Now we have the values - compute the reduction in noise in the baseline period
    ################################################################################
    keywords = ['snr_med', 'snr_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/rms_baseline.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        res_med_prep = infile[keywords[0]][()]
        res_tib_prep = infile[keywords[1]][()]

    # PCA
    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/rms_baseline.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        res_med_pca = infile[keywords[0]][()]
        res_tib_pca = infile[keywords[1]][()]

    # Changing to mean of means after ratio is already calculated
    residual_med_pca = (np.mean(res_med_pca / res_med_prep, axis=tuple([0, 1]))) * 100
    residual_tib_pca = (np.mean(res_tib_pca / res_tib_prep, axis=tuple([0, 1]))) * 100

    print(f"Residual PCA Medial: {residual_med_pca:.4f}%")
    print(f"Residual PCA Tibial: {residual_tib_pca:.4f}%")

    # ICA
    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/rms_baseline.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        res_med_ica = infile[keywords[0]][()]
        res_tib_ica = infile[keywords[1]][()]

    residual_med_ica = (np.mean(res_med_ica / res_med_prep, axis=tuple([0, 1]))) * 100
    residual_tib_ica = (np.mean(res_tib_ica / res_tib_prep, axis=tuple([0, 1]))) * 100

    print(f"Residual ICA Medial: {residual_med_ica:.4f}%")
    print(f"Residual ICA Tibial: {residual_tib_ica:.4f}%")

    # SSP
    for n in np.arange(5, 21):
        fn = f"/data/pt_02569/tmp_data/ssp_py/rms_baseline_{n}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            res_med_ssp = infile[keywords[0]][()]
            res_tib_ssp = infile[keywords[1]][()]

        residual_med_ssp = (np.mean(res_med_ssp / res_med_prep, axis=tuple([0, 1]))) * 100
        residual_tib_ssp = (np.mean(res_tib_ssp / res_tib_prep, axis=tuple([0, 1]))) * 100

        print(f"Residual SSP Medial {n}: {residual_med_ssp:.4f}%")
        print(f"Residual SSP Tibial {n}: {residual_tib_ssp:.4f}%")

    ############################################################################################
    # Now look at residual intensity for just our channels of interest
    #     if cond_name == 'tibial':
    #         channels = ['S23', 'L1', 'S31']
    #     elif cond_name == 'median':
    #         channels = ['S6', 'SC6', 'S14']
    ###########################################################################################
    print('\n')
    median_pos = []
    tibial_pos = []
    for channel in ['S23', 'L1', 'S31']:
        tibial_pos.append(esg_chans.index(channel))
    for channel in ['S6', 'SC6', 'S14']:
        median_pos.append(esg_chans.index(channel))

    # All files are 36x39 dimensions - n_subjects x n_channels
    keywords = ['snr_med', 'snr_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/rms_baseline.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        res_med_prep = infile[keywords[0]][()]
        res_tib_prep = infile[keywords[1]][()]

    # PCA
    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/rms_baseline.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        res_med_pca = infile[keywords[0]][()]
        res_tib_pca = infile[keywords[1]][()]

    residual_med_pca = (np.mean(res_med_pca[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    residual_tib_pca = (np.mean(res_tib_pca[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100

    print(f"Residual PCA Medial: {residual_med_pca:.4f}%")
    print(f"Residual PCA Tibial: {residual_tib_pca:.4f}%")

    # ICA
    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/rms_baseline.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        res_med_ica = infile[keywords[0]][()]
        res_tib_ica = infile[keywords[1]][()]

    residual_med_ica = (np.mean(res_med_ica[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    residual_tib_ica = (np.mean(res_tib_ica[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100

    print(f"Residual ICA Medial: {residual_med_ica:.4f}%")
    print(f"Residual ICA Tibial: {residual_tib_ica:.4f}%")

    # SSP
    for n in np.arange(5, 21):
        fn = f"/data/pt_02569/tmp_data/ssp_py/rms_baseline_{n}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            res_med_ssp = infile[keywords[0]][()]
            res_tib_ssp = infile[keywords[1]][()]

        residual_med_ssp = (np.mean(res_med_ssp[:, median_pos] / res_med_prep[:, median_pos],
                                    axis=tuple([0, 1]))) * 100
        residual_tib_ssp = (np.mean(res_tib_ssp[:, tibial_pos] / res_tib_prep[:, tibial_pos],
                                    axis=tuple([0, 1]))) * 100

        print(f"Residual SSP Medial {n}: {residual_med_ssp:.4f}%")
        print(f"Residual SSP Tibial {n}: {residual_tib_ssp:.4f}%")