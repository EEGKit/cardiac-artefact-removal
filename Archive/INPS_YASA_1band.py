# File to compute the inps in a different way - using the bandpower functionality of the yasa package
# Will just compute 1 band, either f_0 +5, or f_0 +10

from scipy.io import loadmat
import numpy as np
import mne
import yasa
import h5py


def get_harmonics(raw1, trigger, sample_rate):
    minutes = (raw1.times[-1])/60  # Gets end time in seconds, convert to minutes
    events, event_ids = mne.events_from_annotations(raw1)
    relevant_events = mne.pick_events(events, include=event_ids[trigger])
    heart_rate = max(np.shape(relevant_events))/minutes  # This gives the number of R-peaks divided by time in min
    freqs = []
    f_0 = heart_rate/60
    f_0 = np.around(f_0, decimals=1)
    freqs.append(f_0)
    for nq in np.arange(2, 6):
        freqs.append(nq*f_0)

    return freqs


if __name__ == '__main__':

    calc_prepared = False
    calc_PCA = False
    calc_ICA = False
    choose_limited = False  # If true use ICA data with top 4 components chosen - use FALSE, see main
    calc_SSP = False
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

    for band in [5, 10]:
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
                    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
                    raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif",
                                              preload=True)

                    freq = get_harmonics(raw, trigger_name, sampling_rate)

                    # Compute power at the harmonics
                    # Rounding to 1 decimal place - want to keep frequency resolution as high as possible
                    freq = np.around(freq, decimals=1)
                    data = raw.pick_channels(esg_chans).reorder_channels(esg_chans)._data * 1e6
                    # PSD will have units uV^2 now
                    # Need a frequency resolution of 0.1Hz - win set to 10 seconds
                    # This outputs a dataframe
                    bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
                                        bandpass=False,
                                        bands=[(freq[0], freq[0]+band, 'f0')],
                                        kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})

                    # Extract the absolute power from the relative powers output above
                    bands = ['f0']
                    bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
                    bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
                    power = bp_abs['Sum'].values  # Extract value of interest

                    # Now have power for each subject, insert it into the correct condition array
                    if cond_name == 'median':
                        pow_med_prep[subject - 1, :] = power
                    elif cond_name == 'tibial':
                        pow_tib_prep[subject - 1, :] = power

            # Save to file
            savepow.pow_med = pow_med_prep
            savepow.pow_tib = pow_tib_prep
            dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]

            fn = f"/data/pt_02569/tmp_data/prepared_py/inps_yasa_{band}.h5"

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
                    input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id
                    fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                    raw = mne.io.read_raw_fif(input_path+fname, preload=True)

                    freq = get_harmonics(raw, trigger_name, sampling_rate)

                    # Now we have the raw data in the filtered form
                    # We have the fundamental frequency and harmonics for this subject
                    # Compute power at the frequencies
                    freq = np.around(freq, decimals=1)

                    data = raw.pick_channels(esg_chans).reorder_channels(esg_chans)._data * 1e6
                    # PSD will have units uV^2 now
                    # Need a frequency resolution of 0.1Hz - win set to 10 seconds
                    # This outputs a dataframe
                    bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
                                        bandpass=False,
                                        bands=[(freq[0], freq[0] + band, 'f0')],
                                        kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})

                    # Extract the absolute power from the relative powers output above
                    bands = ['f0']
                    bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
                    bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
                    power = bp_abs['Sum'].values  # Extract value of interest

                    # Now have power for each subject, insert it into the correct condition array
                    if cond_name == 'median':
                        pow_med_pca[subject - 1, :] = power
                    elif cond_name == 'tibial':
                        pow_tib_pca[subject - 1, :] = power

            # Save to file
            savepow.pow_med = pow_med_pca
            savepow.pow_tib = pow_tib_pca
            dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]

            fn = f"/data/pt_02569/tmp_data/ecg_rm_py/inps_yasa_{band}.h5"

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
                    input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id
                    if choose_limited:
                        fname = f"clean_baseline_ica_auto_{cond_name}_lim.fif"
                    else:
                        fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                    freq = get_harmonics(raw, trigger_name, sampling_rate)

                    # Now we have the raw data in the filtered form
                    # We have the fundamental frequency and harmonics for this subject
                    # Compute power at the frequencies
                    # freq = np.around(freq, decimals=1)
                    # data = raw.get_data(esg_chans) * 1e6  # Both give same answer
                    freq = np.around(freq, decimals=1)

                    data = raw.pick_channels(esg_chans).reorder_channels(esg_chans)._data * 1e6
                    # PSD will have units uV^2 now
                    # Need a frequency resolution of 0.1Hz - win set to 10 seconds
                    # This outputs a dataframe
                    bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
                                        bandpass=False,
                                        bands=[(freq[0], freq[0] + band, 'f0')],
                                        kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})

                    # Extract the absolute power from the relative powers output above
                    bands = ['f0']
                    bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
                    bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
                    power = bp_abs['Sum'].values  # Extract value of interest

                    # Now have power for each subject, insert it into the correct condition array
                    if cond_name == 'median':
                        pow_med_ica[subject - 1, :] = power
                    elif cond_name == 'tibial':
                        pow_tib_ica[subject - 1, :] = power

            # Save to file
            savepow.pow_med = pow_med_ica
            savepow.pow_tib = pow_tib_ica
            dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]

            if choose_limited:
                fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa_lim_{band}.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa_{band}.h5"

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

            for n in np.arange(5, 7):
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
                        input_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id
                        savename = input_path + "/" + str(n) + " projections/"
                        raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")

                        freq = get_harmonics(raw, trigger_name, sampling_rate)

                        # Now we have the raw data in the filtered form
                        # We have the fundamental frequency and harmonics for this subject
                        # Compute power at the frequencies
                        freq = np.around(freq, decimals=1)
                        data = raw.get_data(esg_chans) * 1e6

                        # PSD will have units uV^2 now
                        # Need a frequency resolution of 0.1Hz - win set to 10 seconds
                        # This outputs a dataframe
                        bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
                                            bandpass=False,
                                            bands=[(freq[0], freq[0] + band, 'f0')],
                                            kwargs_welch={'scaling': 'spectrum', 'average': 'median',
                                                          'window': 'hamming'})

                        # Extract the absolute power from the relative powers output above
                        bands = ['f0']
                        bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
                        bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
                        power = bp_abs['Sum'].values  # Extract value of interest

                        # Now have power for each subject, insert it into the correct condition array
                        if cond_name == 'median':
                            pow_med_ssp[subject - 1, :] = power
                        elif cond_name == 'tibial':
                            pow_tib_ssp[subject - 1, :] = power

                # Save to file
                savepow.pow_med = pow_med_ssp
                savepow.pow_tib = pow_tib_ssp
                dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]

                fn = f"/data/pt_02569/tmp_data/ssp_py/inps_yasa_{n}_{band}.h5"

                with h5py.File(fn, "w") as outfile:
                    for keyword in dataset_keywords:
                        outfile.create_dataset(keyword, data=getattr(savepow, keyword))

    ##########################################################################
    # Calculate INPS for each - Prepared divided by clean
    ##########################################################################
    for band in [5, 10]:
        print(f'All channels, band {band}')

        # All being read in have have n_subjects x n_channels (36, 39)
        keywords = ['pow_med', 'pow_tib']
        fn = f"/data/pt_02569/tmp_data/prepared_py/inps_yasa_{band}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            pow_med_prep = infile[keywords[0]][()]
            pow_tib_prep = infile[keywords[1]][()]

        # PCA
        fn = f"/data/pt_02569/tmp_data/ecg_rm_py/inps_yasa_{band}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            pow_med_pca = infile[keywords[0]][()]
            pow_tib_pca = infile[keywords[1]][()]

        # Changing to mean of means after ratio is already calculated
        residual_med_pca = (np.mean(pow_med_prep / pow_med_pca, axis=tuple([0, 1])))
        residual_tib_pca = (np.mean(pow_tib_prep / pow_tib_pca, axis=tuple([0, 1])))

        print(f"Residual PCA Medial: {residual_med_pca:.4e}")
        print(f"Residual PCA Tibial: {residual_tib_pca:.4e}")

        # ICA
        if choose_limited:
            fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa_lim_{band}.h5"
        else:
            fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa_{band}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            pow_med_ica = infile[keywords[0]][()]
            pow_tib_ica = infile[keywords[1]][()]

        # Changing to mean of means after ratio is already calculated
        residual_med_ica = (np.mean(pow_med_prep / pow_med_ica, axis=tuple([0, 1])))
        residual_tib_ica = (np.mean(pow_tib_prep / pow_tib_ica, axis=tuple([0, 1])))

        print(f"Residual ICA Medial: {residual_med_ica:.4e}")
        print(f"Residual ICA Tibial: {residual_tib_ica:.4e}")

        # SSP
        for n in np.arange(5, 7):
            fn = f"/data/pt_02569/tmp_data/ssp_py/inps_yasa_{n}_{band}.h5"
            with h5py.File(fn, "r") as infile:
                # Get the data
                pow_med_ssp = infile[keywords[0]][()]
                pow_tib_ssp = infile[keywords[1]][()]

            # Changing to mean of means after ratio is already calculated
            residual_med_ssp = (np.mean(pow_med_prep / pow_med_ssp, axis=tuple([0, 1])))
            residual_tib_ssp = (np.mean(pow_tib_prep / pow_tib_ssp, axis=tuple([0, 1])))

            print(f"Residual SSP Medial {n}: {residual_med_ssp:.4e}")
            print(f"Residual SSP Tibial {n}: {residual_tib_ssp:.4e}")

    ############################################################################################
    # Now look at INPS for just our channels of interest
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

    for band in [5, 10]:
        print(f"3 relevant channels, band {band}")
        # All files are 36x39 dimensions - n_subjects x n_channels
        keywords = ['pow_med', 'pow_tib']
        fn = f"/data/pt_02569/tmp_data/prepared_py/inps_yasa_{band}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            pow_med_prep = infile[keywords[0]][()]
            pow_tib_prep = infile[keywords[1]][()]

        # PCA
        fn = f"/data/pt_02569/tmp_data/ecg_rm_py/inps_yasa_{band}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            pow_med_pca = infile[keywords[0]][()]
            pow_tib_pca = infile[keywords[1]][()]

        residual_med_pca = (np.mean(pow_med_prep[:, median_pos] / pow_med_pca[:, median_pos], axis=tuple([0, 1])))
        residual_tib_pca = (np.mean(pow_tib_prep[:, tibial_pos] / pow_tib_pca[:, tibial_pos], axis=tuple([0, 1])))

        print(f"Residual PCA Medial: {residual_med_pca:.4e}")
        print(f"Residual PCA Tibial: {residual_tib_pca:.4e}")

        # ICA
        if choose_limited:
            fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa_lim_{band}.h5"
        else:
            fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa_{band}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            pow_med_ica = infile[keywords[0]][()]
            pow_tib_ica = infile[keywords[1]][()]

        residual_med_ica = (np.mean(pow_med_prep[:, median_pos] / pow_med_ica[:, median_pos], axis=tuple([0, 1])))
        residual_tib_ica = (np.mean(pow_tib_prep[:, tibial_pos] / pow_tib_ica[:, tibial_pos], axis=tuple([0, 1])))

        print(f"Residual ICA Medial: {residual_med_ica:.4e}")
        print(f"Residual ICA Tibial: {residual_tib_ica:.4e}")

        # SSP
        for n in np.arange(5, 7):
            fn = f"/data/pt_02569/tmp_data/ssp_py/inps_yasa_{n}_{band}.h5"
            with h5py.File(fn, "r") as infile:
                # Get the data
                pow_med_ssp = infile[keywords[0]][()]
                pow_tib_ssp = infile[keywords[1]][()]

            residual_med_ssp = (np.mean(pow_med_prep[:, median_pos] / pow_med_ssp[:, median_pos],axis=tuple([0, 1])))
            residual_tib_ssp = (np.mean(pow_tib_prep[:, tibial_pos] / pow_tib_ssp[:, tibial_pos],axis=tuple([0, 1])))

            print(f"Residual SSP Medial {n}: {residual_med_ssp:.4e}")
            print(f"Residual SSP Tibial {n}: {residual_tib_ssp:.4e}")
