# Script to examine the standard deviation of the data about the time points at which fitting ends in PCA
# Jumps noticed in the plots - want to quantify their effect

import mne
import numpy as np
import h5py
from scipy.io import loadmat

if __name__ == '__main__':
    # Set which to run
    get_timings = False
    calc_raw = True
    calc_PCA = True
    calc_ICA = True
    calc_SSP = True

    # Testing with just subject 1 at the moment
    subjects = np.arange(1, 37)  # (1, 2) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    ##########################################################
    # Get timings of the fit_end for each subject and condition
    ###########################################################
    if get_timings:
        for subject in subjects:
            for cond_name in cond_names:
                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from preparation script
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Extract relevant events
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == 'fit_end'}
                events = events[np.where(events[:, 2] == event_id_dict['fit_end'])]
                event_times = events[:, 0]

                # Save each array related to subject + condition to a h5 file
                if cond_name == 'median':
                    fn = f"/data/pt_02569/tmp_data/fit_end_timings/fit_end_timings_{cond_name}_{subject}.h5"
                    with h5py.File(fn, 'w') as hf:
                        hf.create_dataset(f'timings', data=event_times)
                elif cond_name == 'tibial':
                    fn = f"/data/pt_02569/tmp_data/fit_end_timings/fit_end_timings_{cond_name}_{subject}.h5"
                    with h5py.File(fn, 'w') as hf:
                        hf.create_dataset(f'timings', data=event_times)

    #########################################
    # Raw Calculations
    #########################################
    if calc_raw:
        # Declare class to hold ecg fit information
        class save_STD():
            def __init__(self):
                pass

        # Instantiate class
        savestd = save_STD()

        # Matrix of dimensions no.subjects x no. channels
        std_med = np.zeros((len(subjects), 39))
        std_tib = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from preparation script
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                fname = f"noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Load the fit_end times relevant here - is just a list of sample points
                fn = f"/data/pt_02569/tmp_data/fit_end_timings/fit_end_timings_{cond_name}_{subject}.h5"
                with h5py.File(fn, "r") as infile:
                    # Get the data
                    fit_end = infile['timings'][()].reshape(-1)

                # add reference channel to data
                mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                # Want to get the std of the data in a 20ms period about each fit end timing
                # get_data returns shape (n_channels, n_times)
                # At the finish, std will be of size (len(fit_end), n_channels)
                raw.reorder_channels(esg_chans)
                std = []
                for point in fit_end:
                    data = raw.get_data(picks=esg_chans, start=int(point-(10/1000)*sampling_rate),
                                        stop=int(point+(10/1000)*sampling_rate))
                    std.append(np.std(data, axis=1))

                # Then want the average standard deviation for each channel - will give shape (n_channels, )
                stan_dev = np.mean(std, axis=0)

                # Now have std related to each subject (39 channels) and condition
                if cond_name == 'median':
                    std_med[subject - 1, :] = stan_dev
                elif cond_name == 'tibial':
                    std_tib[subject - 1, :] = stan_dev

        # Save to file to compare to matlab - only for debugging
        savestd.snr_med = std_med
        savestd.snr_tib = std_tib
        dataset_keywords = [a for a in dir(savestd) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/prepared_py/std.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savestd, keyword))

    #########################################
    # PCA Calculations
    #########################################
    if calc_PCA:
        # Declare class to hold ecg fit information
        class save_STD():
            def __init__(self):
                pass

        # Instantiate class
        savestd = save_STD()

        # Matrix of dimensions no.subjects x no. channels
        std_med = np.zeros((len(subjects), 39))
        std_tib = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from preparation script
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Load the fit_end times relevant here - is just a list of sample points
                fn = f"/data/pt_02569/tmp_data/fit_end_timings/fit_end_timings_{cond_name}_{subject}.h5"
                with h5py.File(fn, "r") as infile:
                    # Get the data
                    fit_end = infile['timings'][()].reshape(-1)

                # add reference channel to data
                mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                # Want to get the std of the data in a 20ms period about each fit end timing
                # get_data returns shape (n_channels, n_times)
                # At the finish, std will be of size (len(fit_end), n_channels)
                raw.reorder_channels(esg_chans)
                std = []
                for point in fit_end:
                    data = raw.get_data(picks=esg_chans, start=int(point - (10 / 1000) * sampling_rate),
                                        stop=int(point + (10 / 1000) * sampling_rate))
                    std.append(np.std(data, axis=1))

                # Then want the average standard deviation for each channel - will give shape (n_channels, )
                stan_dev = np.mean(std, axis=0)

                # Now have std related to each subject (39 channels) and condition
                if cond_name == 'median':
                    std_med[subject - 1, :] = stan_dev
                elif cond_name == 'tibial':
                    std_tib[subject - 1, :] = stan_dev

        # Save to file to compare to matlab - only for debugging
        savestd.snr_med = std_med
        savestd.snr_tib = std_tib
        dataset_keywords = [a for a in dir(savestd) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/ecg_rm_py/std.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savestd, keyword))

    #########################################
    # ICA Calculations
    #########################################
    if calc_ICA:
        # Declare class to hold ecg fit information
        class save_STD():
            def __init__(self):
                pass

        # Instantiate class
        savestd = save_STD()

        # Matrix of dimensions no.subjects x no. channels
        std_med = np.zeros((len(subjects), 39))
        std_tib = np.zeros((len(subjects), 39))

        for subject in subjects:
            for cond_name in cond_names:

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from preparation script
                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Load the fit_end times relevant here - is just a list of sample points
                fn = f"/data/pt_02569/tmp_data/fit_end_timings/fit_end_timings_{cond_name}_{subject}.h5"
                with h5py.File(fn, "r") as infile:
                    # Get the data
                    fit_end = infile['timings'][()].reshape(-1)

                # Want to get the std of the data in a 20ms period about each fit end timing
                # get_data returns shape (n_channels, n_times)
                # At the finish, std will be of size (len(fit_end), n_channels)
                raw.reorder_channels(esg_chans)
                std = []
                for point in fit_end:
                    data = raw.get_data(picks=esg_chans, start=int(point - (10 / 1000) * sampling_rate),
                                        stop=int(point + (10 / 1000) * sampling_rate))
                    std.append(np.std(data, axis=1))

                # Then want the average standard deviation for each channel - will give shape (n_channels, )
                stan_dev = np.mean(std, axis=0)

                # Now have std related to each subject (39 channels) and condition
                if cond_name == 'median':
                    std_med[subject - 1, :] = stan_dev
                elif cond_name == 'tibial':
                    std_tib[subject - 1, :] = stan_dev

        # Save to file to compare to matlab - only for debugging
        savestd.snr_med = std_med
        savestd.snr_tib = std_tib
        dataset_keywords = [a for a in dir(savestd) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/baseline_ica_py/std.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savestd, keyword))

    #########################################
    # SSP Calculations
    #########################################
    if calc_SSP:
        # Declare class to hold ecg fit information
        class save_STD():
            def __init__(self):
                pass

        # Instantiate class
        savestd = save_STD()

        for n in np.arange(5, 21):
            # Matrix of dimensions no.subjects x no. channels
            std_med = np.zeros((len(subjects), 39))
            std_tib = np.zeros((len(subjects), 39))

            for subject in subjects:
                for cond_name in cond_names:

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Want the SNR
                    # Load data resulting from preparation script
                    input_path = "/data/p_02569/SSP/" + subject_id
                    savename = input_path + "/" + str(n) + " projections/"
                    raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")

                    # Load the fit_end times relevant here - is just a list of sample points
                    fn = f"/data/pt_02569/tmp_data/fit_end_timings/fit_end_timings_{cond_name}_{subject}.h5"
                    with h5py.File(fn, "r") as infile:
                        # Get the data
                        fit_end = infile['timings'][()].reshape(-1)

                    # Want to get the std of the data in a 20ms period about each fit end timing
                    # get_data returns shape (n_channels, n_times)
                    # At the finish, std will be of size (len(fit_end), n_channels)
                    raw.reorder_channels(esg_chans)
                    std = []
                    for point in fit_end:
                        data = raw.get_data(picks=esg_chans, start=int(point - (10 / 1000) * sampling_rate),
                                            stop=int(point + (10 / 1000) * sampling_rate))
                        std.append(np.std(data, axis=1))

                    # Then want the average standard deviation for each channel - will give shape (n_channels, )
                    stan_dev = np.mean(std, axis=0)

                    # Now have std related to each subject (39 channels) and condition
                    if cond_name == 'median':
                        std_med[subject - 1, :] = stan_dev
                    elif cond_name == 'tibial':
                        std_tib[subject - 1, :] = stan_dev

            # Save to file to compare to matlab - only for debugging
            savestd.snr_med = std_med
            savestd.snr_tib = std_tib
            dataset_keywords = [a for a in dir(savestd) if not a.startswith('__')]

            fn = f"/data/p_02569/SSP/std_{n}.h5"

            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savestd, keyword))

    #########################################################
    # Now we have all the std values, print to screen
    #########################################################
    keywords = ['snr_med', 'snr_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/std.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_prep = infile[keywords[0]][()]
        std_tib_prep = infile[keywords[1]][()]

    print(f"STD Prep Medial: {np.mean(std_med_prep, axis=tuple([0, 1])):.4e}")
    print(f"STD Prep Tibial: {np.mean(std_tib_prep, axis=tuple([0, 1])):.4e}")

    # PCA
    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/std.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_pca = infile[keywords[0]][()]
        std_tib_pca = infile[keywords[1]][()]

    print(f"STD PCA Medial: {np.mean(std_med_pca, axis=tuple([0, 1])):.4e}")
    print(f"STD PCA Tibial: {np.mean(std_tib_pca, axis=tuple([0, 1])):.4e}")

    # ICA
    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/std.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_ica = infile[keywords[0]][()]
        std_tib_ica = infile[keywords[1]][()]

    print(f"STD ICA Medial: {np.mean(std_med_ica, axis=tuple([0, 1])):.4e}")
    print(f"STD ICA Tibial: {np.mean(std_tib_ica, axis=tuple([0, 1])):.4e}")

    # SSP
    for n in np.arange(5, 21):
        fn = f"/data/p_02569/SSP/std_{n}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            std_med_ssp = infile[keywords[0]][()]
            std_tib_ssp = infile[keywords[1]][()]

        print(f"STD SSP Medial {n}: {np.mean(std_med_ssp, axis=tuple([0, 1])):.4e}")
        print(f"STD SSP Tibial {n}: {np.mean(std_tib_ssp, axis=tuple([0, 1])):.4e}")

    #################################################################################
    # STD only in channels of interest
    #################################################################################
    print('\n')
    median_pos = []
    tibial_pos = []
    for channel in ['S23', 'L1', 'S31']:
        tibial_pos.append(esg_chans.index(channel))
    for channel in ['S6', 'SC6', 'S14']:
        median_pos.append(esg_chans.index(channel))

    keywords = ['snr_med', 'snr_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/std.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_prep = infile[keywords[0]][()]
        std_tib_prep = infile[keywords[1]][()]

    print(f"STD Prep Medial: {np.mean(std_med_prep[:, median_pos], axis=tuple([0, 1])):.4e}")
    print(f"STD Prep Tibial: {np.mean(std_tib_prep[:, tibial_pos], axis=tuple([0, 1])):.4e}")

    # PCA
    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/std.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_pca = infile[keywords[0]][()]
        std_tib_pca = infile[keywords[1]][()]

    print(f"STD PCA Medial: {np.mean(std_med_pca[:, median_pos], axis=tuple([0, 1])):.4e}")
    print(f"STD PCA Tibial: {np.mean(std_tib_pca[:, tibial_pos], axis=tuple([0, 1])):.4e}")

    # ICA
    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/std.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_ica = infile[keywords[0]][()]
        std_tib_ica = infile[keywords[1]][()]

    print(f"STD ICA Medial: {np.mean(std_med_ica[:, median_pos], axis=tuple([0, 1])):.4e}")
    print(f"STD ICA Tibial: {np.mean(std_tib_ica[:, tibial_pos], axis=tuple([0, 1])):.4e}")

    # SSP
    for n in np.arange(5, 21):
        fn = f"/data/p_02569/SSP/std_{n}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            std_med_ssp = infile[keywords[0]][()]
            std_tib_ssp = infile[keywords[1]][()]

        print(f"STD SSP Medial {n}: {np.mean(std_med_ssp[:, median_pos], axis=tuple([0, 1])):.4e}")
        print(f"STD SSP Tibial {n}: {np.mean(std_tib_ssp[:, tibial_pos], axis=tuple([0, 1])):.4e}")

    ########################################################################
    # STD across all channels expressed as percentage change
    ########################################################################
    print('\n')
    keywords = ['snr_med', 'snr_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/std.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_prep = infile[keywords[0]][()]
        std_tib_prep = infile[keywords[1]][()]

    # PCA
    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/std.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_pca = infile[keywords[0]][()]
        std_tib_pca = infile[keywords[1]][()]

    print(f"STD PCA Medial: {np.mean((std_med_pca-std_med_prep)/std_med_prep, axis=tuple([0, 1]))*100:.4f}%")
    print(f"STD PCA Tibial: {np.mean((std_tib_pca-std_tib_prep)/std_tib_prep, axis=tuple([0, 1]))*100:.4f}%")

    # ICA
    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/std.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_ica = infile[keywords[0]][()]
        std_tib_ica = infile[keywords[1]][()]

    print(f"STD ICA Medial: {np.mean((std_med_ica-std_med_prep)/std_med_prep, axis=tuple([0, 1]))*100:.4f}%")
    print(f"STD ICA Tibial: {np.mean((std_tib_ica-std_tib_prep)/std_tib_prep, axis=tuple([0, 1]))*100:.4f}%")

    # SSP
    for n in np.arange(5, 21):
        fn = f"/data/p_02569/SSP/std_{n}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            std_med_ssp = infile[keywords[0]][()]
            std_tib_ssp = infile[keywords[1]][()]

        print(f"STD SSP Medial {n}: {np.mean((std_med_ssp-std_med_prep)/std_med_prep, axis=tuple([0, 1]))*100:.4f}%")
        print(f"STD SSP Tibial {n}: {np.mean((std_tib_ssp-std_tib_prep)/std_tib_prep, axis=tuple([0, 1]))*100:.4f}%")
