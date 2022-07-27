# File to compute the inps using the bandpower functionality of the yasa package

from scipy.io import loadmat
import numpy as np
import mne
import yasa
import h5py


# Function to get the fundamental frequency of the heartbeat and the first 4 harmonics
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


# Function to get the power at the fundamental frequency and the first 4 harmonics
def get_power(data1, freq1, sampling_rate1, esg_channels):
    # PSD will have units uV^2 now
    # Need a frequency resolution of 0.1Hz - win set to 10 seconds
    # This outputs a dataframe
    bp = yasa.bandpower(data1, sf=sampling_rate1, ch_names=esg_channels, win_sec=10, relative=True,
                        bandpass=False,
                        bands=[(freq1[0] - 0.1, freq1[0] + 0.1, 'f0'),
                               (freq1[1] - 0.1, freq1[1] + 0.1, 'f1'),
                               (freq1[2] - 0.1, freq1[2] + 0.1, 'f2'),
                               (freq1[3] - 0.1, freq1[3] + 0.1, 'f3'),
                               (freq1[4] - 0.1, freq1[4] + 0.1, 'f4')],
                        kwargs_welch={'scaling': 'spectrum', 'average': 'median',
                                      'window': 'hamming'})

    # Extract the absolute power from the relative powers output above
    bands = ['f0', 'f1', 'f2', 'f3', 'f4']
    bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
    bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
    full_power = bp_abs['Sum'].values  # Extract value of interest

    return full_power


if __name__ == '__main__':
    choose_limited = False  # If true use ICA data with top 4 components chosen
    reduced_epochs = False  # Always false in this script as I don't reduce epochs

    # Define the channel names so they come out of each dataset the same
    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')

    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    # Loop through methods and save as required
    which_method = {'Prep': True,
                    'PCA': True,
                    'PCA PCHIP': True,
                    'PCA Tukey': True,
                    'PCA Tukey PCHIP': True,
                    'ICA': True,
                    'Post-ICA': True,
                    'SSP': True}

    for i in np.arange(0, len(which_method)):
        method = list(which_method.keys())[i]
        if which_method[method]:  # If this method is true, go through with the rest
            class save_pow():
                def __init__(self):
                    pass

            # Instantiate class
            savepow = save_pow()

            if method == 'SSP':
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

                            # Load data
                            input_path = "/data/p_02569/SSP/" + subject_id
                            savename = input_path + "/" + str(n) + " projections/"
                            raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")

                            # Compute power at the fundamental frequency and harmonics
                            freq = get_harmonics(raw, trigger_name, sampling_rate)
                            freq = np.around(freq, decimals=1)
                            data = raw.get_data(esg_chans) * 1e6
                            power = get_power(data, freq, sampling_rate, esg_chans)

                            # Insert power for each subject&condition
                            if cond_name == 'median':
                                pow_med_ssp[subject - 1, :] = power
                            elif cond_name == 'tibial':
                                pow_tib_ssp[subject - 1, :] = power

                    # Save to file
                    savepow.pow_med = pow_med_ssp
                    savepow.pow_tib = pow_tib_ssp
                    dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]
                    fn = f"/data/p_02569/SSP/inps_yasa_{n}.h5"
                    with h5py.File(fn, "w") as outfile:
                        for keyword in dataset_keywords:
                            outfile.create_dataset(keyword, data=getattr(savepow, keyword))

            else:
                # Matrix of dimensions no.subjects x no_channels
                pow_med = np.zeros((len(subjects), 39))
                pow_tib = np.zeros((len(subjects), 39))
                for subject in subjects:
                    for cond_name in cond_names:
                        if cond_name == 'tibial':
                            trigger_name = 'qrs'
                            nerve = 2
                        elif cond_name == 'median':
                            trigger_name = 'qrs'
                            nerve = 1

                        subject_id = f'sub-{str(subject).zfill(3)}'

                        # Get the right file path
                        if method == 'Prep':
                            file_path = "/data/pt_02569/tmp_data/prepared_py/"
                            file_name = f'noStimart_sr1000_{cond_name}_withqrs.fif'
                        elif method == 'PCA':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                        elif method == 'PCA PCHIP':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif'
                        elif method == 'PCA Tukey':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                        elif method == 'PCA Tukey PCHIP':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif'
                        elif method == 'ICA':
                            file_path = "/data/pt_02569/tmp_data/baseline_ica_py/"
                            if choose_limited:
                                file_name = f'clean_baseline_ica_auto_{cond_name}_lim.fif'
                            else:
                                file_name = f'clean_baseline_ica_auto_{cond_name}.fif'
                        elif method == 'Post-ICA':
                            file_path = "/data/pt_02569/tmp_data/ica_py/"
                            file_name = f'clean_ica_auto_{cond_name}.fif'

                        input_path = file_path + subject_id + "/esg/prepro/"
                        raw = mne.io.read_raw_fif(f"{input_path}{file_name}", preload=True)

                        if (method == 'Prep' or method == 'PCA' or method == 'PCA Tukey' or method == 'PCA PCHIP' or
                                method == 'PCA Tukey PCHIP'):
                            # add reference channel to data
                            mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
                            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                                       method='iir',
                                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                        # Compute power at the harmonics
                        freq = get_harmonics(raw, trigger_name, sampling_rate)
                        freq = np.around(freq, decimals=1)
                        data = raw.pick_channels(esg_chans).reorder_channels(esg_chans)._data * 1e6
                        power = get_power(data, freq, sampling_rate, esg_chans)

                        # Insert power for subject&condition
                        if cond_name == 'median':
                            pow_med[subject - 1, :] = power
                        elif cond_name == 'tibial':
                            pow_tib[subject - 1, :] = power

                # Save to file
                savepow.pow_med = pow_med
                savepow.pow_tib = pow_tib
                dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]
                if method == 'ICA' and choose_limited:
                    fn = f"{file_path}inps_yasa_lim.h5"
                elif method == 'PCA PCHIP' or method == 'PCA Tukey PCHIP':
                    fn = f"{file_path}inps_yasa_pchip.h5"
                else:
                    fn = f"{file_path}inps_yasa.h5"
                with h5py.File(fn, "w") as outfile:
                    for keyword in dataset_keywords:
                        outfile.create_dataset(keyword, data=getattr(savepow, keyword))

    # Print to Screen #
    ##########################################################################
    # Calculate INPS for each - Prepared divided by clean
    ##########################################################################
    input_paths = {'PCA': "/data/pt_02569/tmp_data/ecg_rm_py/",
                   'PCA PCHIP': "/data/pt_02569/tmp_data/ecg_rm_py/",
                   'PCA Tukey': "/data/pt_02569/tmp_data/ecg_rm_py_tukey/",
                   'PCA Tukey PCHIP': "/data/pt_02569/tmp_data/ecg_rm_py_tukey/",
                   'ICA': "/data/pt_02569/tmp_data/baseline_ica_py/",
                   'Post-ICA': "/data/pt_02569/tmp_data/ica_py/",
                   'SSP': "/data/p_02569/SSP/"}

    # All files are 36x39 dimensions - n_subjects x n_channels
    keywords = ['pow_med', 'pow_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/inps_yasa.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        pow_med_prep = infile[keywords[0]][()]
        pow_tib_prep = infile[keywords[1]][()]

    print("\n")
    print('All Channels Improved Normalised Power Spectrum Ratio')
    for i in np.arange(0, len(input_paths)):
        name = list(input_paths.keys())[i]
        input_path = input_paths[name]

        if name == 'SSP':
            for n in np.arange(5, 21):
                fn = f"/data/p_02569/SSP/inps_yasa_{n}.h5"
                with h5py.File(fn, "r") as infile:
                    # Get the data
                    pow_med = infile[keywords[0]][()]
                    pow_tib = infile[keywords[1]][()]

                inps_med = (np.mean(pow_med_prep / pow_med, axis=tuple([0, 1])))
                inps_tib = (np.mean(pow_tib_prep / pow_tib, axis=tuple([0, 1])))

                print(f"INPS SSP Median {n}: {inps_med:.4e}")
                print(f"INPS SSP Tibial {n}: {inps_tib:.4e}")
        else:
            if name == 'ICA' and choose_limited:
                fn = f"{input_path}inps_yasa_lim.h5"
            elif name == 'PCA Tukey PCHIP' or name == 'PCA PCHIP':
                fn = f"{input_path}inps_yasa_pchip.h5"
            else:
                fn = f"{input_path}inps_yasa.h5"

            with h5py.File(fn, "r") as infile:
                # Get the data
                pow_med = infile[keywords[0]][()]
                pow_tib = infile[keywords[1]][()]

            inps_med = (np.mean(pow_med_prep / pow_med, axis=tuple([0, 1])))
            inps_tib = (np.mean(pow_tib_prep / pow_tib, axis=tuple([0, 1])))

            print(f'INPS {name} Median: {inps_med:.4e}')
            print(f'INPS {name} Tibial: {inps_tib:.4e}')

    ############################################################################################
    # Now look at INPS for just our channels of interest
    #     if cond_name == 'tibial':
    #         channels = ['S23', 'L1', 'S31']
    #     elif cond_name == 'median':
    #         channels = ['S6', 'SC6', 'S14']
    ###########################################################################################
    print('\n')
    print('Relevant Channels Improved Normalised Power Spectrum')
    median_pos = []
    tibial_pos = []
    for channel in ['S23', 'L1', 'S31']:
        tibial_pos.append(esg_chans.index(channel))
    for channel in ['S6', 'SC6', 'S14']:
        median_pos.append(esg_chans.index(channel))

    for i in np.arange(0, len(input_paths)):
        name = list(input_paths.keys())[i]
        input_path = input_paths[name]

        if name == 'SSP':
            for n in np.arange(5, 21):
                fn = f"/data/p_02569/SSP/inps_yasa_{n}.h5"
                with h5py.File(fn, "r") as infile:
                    # Get the data
                    pow_med = infile[keywords[0]][()]
                    pow_tib = infile[keywords[1]][()]

                inps_med = (np.mean(pow_med_prep[:, median_pos] / pow_med[:, median_pos], axis=tuple([0, 1])))
                inps_tib = (np.mean(pow_tib_prep[:, tibial_pos] / pow_tib[:, tibial_pos], axis=tuple([0, 1])))

                print(f"INPS SSP Median {n}: {inps_med:.4e}")
                print(f"INPS SSP Tibial {n}: {inps_tib:.4e}")
        else:
            if name == 'ICA' and choose_limited:
                fn = f"{input_path}inps_yasa_lim.h5"
            elif name == 'PCA Tukey PCHIP' or name == 'PCA PCHIP':
                fn = f"{input_path}inps_yasa_pchip.h5"
            else:
                fn = f"{input_path}inps_yasa.h5"
            with h5py.File(fn, "r") as infile:
                # Get the data
                pow_med = infile[keywords[0]][()]
                pow_tib = infile[keywords[1]][()]
            inps_med = (np.mean(pow_med_prep[:, median_pos] / pow_med[:, median_pos], axis=tuple([0, 1])))
            inps_tib = (np.mean(pow_tib_prep[:, tibial_pos] / pow_tib[:, tibial_pos], axis=tuple([0, 1])))

            print(f'INPS {name} Median: {inps_med:.4e}')
            print(f'INPS {name} Tibial: {inps_tib:.4e}')


    ######################### Old Printing of Relevant Channels #####################
    ######################### All Channels ##########################
    # # All being read in have have n_subjects x n_channels (36, 39)
    # keywords = ['pow_med', 'pow_tib']
    # fn = f"/data/pt_02569/tmp_data/prepared_py/inps_yasa.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_prep = infile[keywords[0]][()]
    #     pow_tib_prep = infile[keywords[1]][()]
    #
    # # PCA
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py/inps_yasa.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_pca = infile[keywords[0]][()]
    #     pow_tib_pca = infile[keywords[1]][()]
    #
    # # Changing to mean of means after ratio is already calculated
    # residual_med_pca = (np.mean(pow_med_prep / pow_med_pca, axis=tuple([0, 1])))
    # residual_tib_pca = (np.mean(pow_tib_prep / pow_tib_pca, axis=tuple([0, 1])))
    #
    # print(f"Residual PCA Medial: {residual_med_pca:.4e}")
    # print(f"Residual PCA Tibial: {residual_tib_pca:.4e}")
    #
    # # PCA PCHIP
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py/inps_yasa_pchip.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_pca = infile[keywords[0]][()]
    #     pow_tib_pca = infile[keywords[1]][()]
    #
    # # Changing to mean of means after ratio is already calculated
    # residual_med_pca = (np.mean(pow_med_prep / pow_med_pca, axis=tuple([0, 1])))
    # residual_tib_pca = (np.mean(pow_tib_prep / pow_tib_pca, axis=tuple([0, 1])))
    #
    # print(f"Residual PCA PCHIP Medial: {residual_med_pca:.4e}")
    # print(f"Residual PCA PCHIP Tibial: {residual_tib_pca:.4e}")
    #
    # # PCA tukey
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/inps_yasa.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_pca = infile[keywords[0]][()]
    #     pow_tib_pca = infile[keywords[1]][()]
    #
    # # Changing to mean of means after ratio is already calculated
    # residual_med_pca = (np.mean(pow_med_prep / pow_med_pca, axis=tuple([0, 1])))
    # residual_tib_pca = (np.mean(pow_tib_prep / pow_tib_pca, axis=tuple([0, 1])))
    #
    # print(f"Residual PCA Tukey Medial: {residual_med_pca:.4e}")
    # print(f"Residual PCA Tukey Tibial: {residual_tib_pca:.4e}")
    #
    # # PCA tukey pchip
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/inps_yasa_pchip.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_pca = infile[keywords[0]][()]
    #     pow_tib_pca = infile[keywords[1]][()]
    #
    # # Changing to mean of means after ratio is already calculated
    # residual_med_pca = (np.mean(pow_med_prep / pow_med_pca, axis=tuple([0, 1])))
    # residual_tib_pca = (np.mean(pow_tib_prep / pow_tib_pca, axis=tuple([0, 1])))
    #
    # print(f"Residual PCA Tukey PCHIP Medial: {residual_med_pca:.4e}")
    # print(f"Residual PCA Tukey PCHIP Tibial: {residual_tib_pca:.4e}")
    #
    # # ICA
    # if choose_limited:
    #     fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa_lim.h5"
    # else:
    #     fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_ica = infile[keywords[0]][()]
    #     pow_tib_ica = infile[keywords[1]][()]
    #
    # # Changing to mean of means after ratio is already calculated
    # residual_med_ica = (np.mean(pow_med_prep / pow_med_ica, axis=tuple([0, 1])))
    # residual_tib_ica = (np.mean(pow_tib_prep / pow_tib_ica, axis=tuple([0, 1])))
    #
    # print(f"Residual ICA Medial: {residual_med_ica:.4e}")
    # print(f"Residual ICA Tibial: {residual_tib_ica:.4e}")
    #
    # # Post ICA
    # fn = f"/data/pt_02569/tmp_data/ica_py/inps_yasa.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_post_ica = infile[keywords[0]][()]
    #     pow_tib_post_ica = infile[keywords[1]][()]
    #
    # # Changing to mean of means after ratio is already calculated
    # residual_med_post_ica = (np.mean(pow_med_prep / pow_med_post_ica, axis=tuple([0, 1])))
    # residual_tib_post_ica = (np.mean(pow_tib_prep / pow_tib_post_ica, axis=tuple([0, 1])))
    #
    # print(f"Residual Post-ICA Medial: {residual_med_post_ica:.4e}")
    # print(f"Residual Post-ICA Tibial: {residual_tib_post_ica:.4e}")
    #
    # # SSP
    # for n in np.arange(5, 21):
    #     if test39:
    #         fn = f"/data/p_02569/SSP/inps_yasa_{n}_39.h5"
    #     else:
    #         fn = f"/data/p_02569/SSP/inps_yasa_{n}.h5"
    #     with h5py.File(fn, "r") as infile:
    #         # Get the data
    #         pow_med_ssp = infile[keywords[0]][()]
    #         pow_tib_ssp = infile[keywords[1]][()]
    #
    #     # Changing to mean of means after ratio is already calculated
    #     residual_med_ssp = (np.mean(pow_med_prep / pow_med_ssp, axis=tuple([0, 1])))
    #     residual_tib_ssp = (np.mean(pow_tib_prep / pow_tib_ssp, axis=tuple([0, 1])))
    #
    #     print(f"Residual SSP Medial {n}: {residual_med_ssp:.4e}")
    #     print(f"Residual SSP Tibial {n}: {residual_tib_ssp:.4e}")
    # ########################## Relevant Channels ###################
    # # Prepared
    # print('\n')
    # median_pos = []
    # tibial_pos = []
    # for channel in ['S23', 'L1', 'S31']:
    #     tibial_pos.append(esg_chans.index(channel))
    # for channel in ['S6', 'SC6', 'S14']:
    #     median_pos.append(esg_chans.index(channel))
    #
    # # All files are 36x39 dimensions - n_subjects x n_channels
    # keywords = ['pow_med', 'pow_tib']
    # fn = f"/data/pt_02569/tmp_data/prepared_py/inps_yasa.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_prep = infile[keywords[0]][()]
    #     pow_tib_prep = infile[keywords[1]][()]
    #
    # # PCA
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py/inps_yasa.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_pca = infile[keywords[0]][()]
    #     pow_tib_pca = infile[keywords[1]][()]
    #
    # residual_med_pca = (np.mean(pow_med_prep[:, median_pos] / pow_med_pca[:, median_pos], axis=tuple([0, 1])))
    # residual_tib_pca = (np.mean(pow_tib_prep[:, tibial_pos] / pow_tib_pca[:, tibial_pos], axis=tuple([0, 1])))
    #
    # print(f"Residual PCA Medial: {residual_med_pca:.4e}")
    # print(f"Residual PCA Tibial: {residual_tib_pca:.4e}")
    #
    # # PCA PCHIP
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py/inps_yasa_pchip.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_pca = infile[keywords[0]][()]
    #     pow_tib_pca = infile[keywords[1]][()]
    #
    # residual_med_pca = (np.mean(pow_med_prep[:, median_pos] / pow_med_pca[:, median_pos], axis=tuple([0, 1])))
    # residual_tib_pca = (np.mean(pow_tib_prep[:, tibial_pos] / pow_tib_pca[:, tibial_pos], axis=tuple([0, 1])))
    #
    # print(f"Residual PCA PCHIP Medial: {residual_med_pca:.4e}")
    # print(f"Residual PCA PCHIP Tibial: {residual_tib_pca:.4e}")
    #
    # # PCA tukey
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/inps_yasa.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_pca = infile[keywords[0]][()]
    #     pow_tib_pca = infile[keywords[1]][()]
    #
    # residual_med_pca = (np.mean(pow_med_prep[:, median_pos] / pow_med_pca[:, median_pos], axis=tuple([0, 1])))
    # residual_tib_pca = (np.mean(pow_tib_prep[:, tibial_pos] / pow_tib_pca[:, tibial_pos], axis=tuple([0, 1])))
    #
    # print(f"Residual PCA Tukey Medial: {residual_med_pca:.4e}")
    # print(f"Residual PCA Tukey Tibial: {residual_tib_pca:.4e}")
    #
    # # PCA tukey pchip
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/inps_yasa_pchip.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_pca = infile[keywords[0]][()]
    #     pow_tib_pca = infile[keywords[1]][()]
    #
    # residual_med_pca = (np.mean(pow_med_prep[:, median_pos] / pow_med_pca[:, median_pos], axis=tuple([0, 1])))
    # residual_tib_pca = (np.mean(pow_tib_prep[:, tibial_pos] / pow_tib_pca[:, tibial_pos], axis=tuple([0, 1])))
    #
    # print(f"Residual PCA Tukey PCHIP Medial: {residual_med_pca:.4e}")
    # print(f"Residual PCA Tukey PCHIP Tibial: {residual_tib_pca:.4e}")
    #
    # # ICA
    # if choose_limited:
    #     fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa_lim.h5"
    # else:
    #     fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_ica = infile[keywords[0]][()]
    #     pow_tib_ica = infile[keywords[1]][()]
    #
    # residual_med_ica = (np.mean(pow_med_prep[:, median_pos] / pow_med_ica[:, median_pos], axis=tuple([0, 1])))
    # residual_tib_ica = (np.mean(pow_tib_prep[:, tibial_pos] / pow_tib_ica[:, tibial_pos], axis=tuple([0, 1])))
    #
    # print(f"Residual ICA Medial: {residual_med_ica:.4e}")
    # print(f"Residual ICA Tibial: {residual_tib_ica:.4e}")
    #
    # # Post-ICA
    # fn = f"/data/pt_02569/tmp_data/ica_py/inps_yasa.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     pow_med_post_ica = infile[keywords[0]][()]
    #     pow_tib_post_ica = infile[keywords[1]][()]
    #
    # residual_med_post_ica = (
    #     np.mean(pow_med_prep[:, median_pos] / pow_med_post_ica[:, median_pos], axis=tuple([0, 1])))
    # residual_tib_post_ica = (
    #     np.mean(pow_tib_prep[:, tibial_pos] / pow_tib_post_ica[:, tibial_pos], axis=tuple([0, 1])))
    #
    # print(f"Residual Post-ICA Medial: {residual_med_post_ica:.4e}")
    # print(f"Residual Post-ICA Tibial: {residual_tib_post_ica:.4e}")
    #
    # # SSP
    # for n in np.arange(5, 21):
    #     fn = f"/data/p_02569/SSP/inps_yasa_{n}.h5"
    #     with h5py.File(fn, "r") as infile:
    #         # Get the data
    #         pow_med_ssp = infile[keywords[0]][()]
    #         pow_tib_ssp = infile[keywords[1]][()]
    #
    #     residual_med_ssp = (np.mean(pow_med_prep[:, median_pos] / pow_med_ssp[:, median_pos], axis=tuple([0, 1])))
    #     residual_tib_ssp = (np.mean(pow_tib_prep[:, tibial_pos] / pow_tib_ssp[:, tibial_pos], axis=tuple([0, 1])))
    #
    #     print(f"Residual SSP Medial {n}: {residual_med_ssp:.4e}")
    #     print(f"Residual SSP Tibial {n}: {residual_tib_ssp:.4e}")

    # ################ Old Code ####################
    # calc_prepared = False
    # calc_PCA = False
    # calc_PCA_pchip = False
    # calc_PCA_tukey = False
    # calc_PCA_tukey_pchip = False
    # calc_post_ICA = False
    # calc_ICA = False
    # calc_SSP = False
    #
    # ##########################################################################
    # # Calculate Power for Prepared Data
    # ##########################################################################
    # if calc_prepared:
    #     class save_pow():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     savepow = save_pow()
    #
    #     # Matrix of dimensions no.subjects x no_channels
    #     pow_med_prep = np.zeros((len(subjects), 39))
    #     pow_tib_prep = np.zeros((len(subjects), 39))
    #
    #     for subject in subjects:
    #         for cond_name in cond_names:
    #             if cond_name == 'tibial':
    #                 trigger_name = 'qrs'
    #                 nerve = 2
    #             elif cond_name == 'median':
    #                 trigger_name = 'qrs'
    #                 nerve = 1
    #
    #             subject_id = f'sub-{str(subject).zfill(3)}'
    #
    #             # Want the RMS of the data
    #             # Load epochs resulting from importing the data - the raw data in this folder has not been rereferenced
    #             input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
    #             raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif",
    #                                       preload=True)
    #
    #             freq = get_harmonics(raw, trigger_name, sampling_rate)
    #
    #             mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
    #
    #             raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
    #                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    #
    #             raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    #
    #             # Compute power at the harmonics
    #             # Rounding to 1 decimal place - want to keep frequency resolution as high as possible
    #             freq = np.around(freq, decimals=1)
    #             data = raw.pick_channels(esg_chans).reorder_channels(esg_chans)._data * 1e6
    #             # PSD will have units uV^2 now
    #             # Need a frequency resolution of 0.1Hz - win set to 10 seconds
    #             # This outputs a dataframe
    #             bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
    #                                 bandpass=False,
    #                                 bands=[(freq[0]-0.1, freq[0]+0.1, 'f0'),
    #                                        (freq[1]-0.1, freq[1]+0.1, 'f1'),
    #                                        (freq[2]-0.1, freq[2]+0.1, 'f2'),
    #                                        (freq[3]-0.1, freq[3]+0.1, 'f3'),
    #                                        (freq[4]-0.1, freq[4]+0.1, 'f4')],
    #                                 kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})
    #
    #             # Extract the absolute power from the relative powers output above
    #             bands = ['f0', 'f1', 'f2', 'f3', 'f4']
    #             bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
    #             bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
    #             power = bp_abs['Sum'].values  # Extract value of interest
    #
    #             # Now have power for each subject, insert it into the correct condition array
    #             if cond_name == 'median':
    #                 pow_med_prep[subject - 1, :] = power
    #             elif cond_name == 'tibial':
    #                 pow_tib_prep[subject - 1, :] = power
    #
    #     # Save to file
    #     savepow.pow_med = pow_med_prep
    #     savepow.pow_tib = pow_tib_prep
    #     dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/prepared_py/inps_yasa.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(savepow, keyword))
    #
    # ##########################################################################
    # # Calculate Power for PCA Data
    # ##########################################################################
    # if calc_PCA:
    #     class save_pow():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     savepow = save_pow()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     pow_med_pca = np.zeros((len(subjects), 39))
    #     pow_tib_pca = np.zeros((len(subjects), 39))
    #
    #     for subject in subjects:
    #         for cond_name in cond_names:
    #             if cond_name == 'tibial':
    #                 trigger_name = 'qrs'
    #                 nerve = 2
    #             elif cond_name == 'median':
    #                 trigger_name = 'qrs'
    #                 nerve = 1
    #
    #             subject_id = f'sub-{str(subject).zfill(3)}'
    #
    #             # Want the RMS of the data
    #             # Load epochs resulting from PCA - the raw data in this folder has not been rereferenced
    #             input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
    #             fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
    #             raw = mne.io.read_raw_fif(input_path+fname, preload=True)
    #
    #             freq = get_harmonics(raw, trigger_name, sampling_rate)
    #
    #             mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
    #
    #             raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
    #                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    #
    #             raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    #
    #             # Now we have the raw data in the filtered form
    #             # We have the fundamental frequency and harmonics for this subject
    #             # Compute power at the frequencies
    #             freq = np.around(freq, decimals=1)
    #
    #             data = raw.pick_channels(esg_chans).reorder_channels(esg_chans)._data * 1e6
    #             # PSD will have units uV^2 now
    #             # Need a frequency resolution of 0.1Hz - win set to 10 seconds
    #             # This outputs a dataframe
    #             # bp1 = yasa.bandpower(raw, win_sec=10, relative=True,
    #             #                      bandpass=False,
    #             #                      bands=[(freq[0] - 0.1, freq[0] + 0.1, 'f0'),
    #             #                             (freq[1] - 0.1, freq[1] + 0.1, 'f1'),
    #             #                             (freq[2] - 0.1, freq[2] + 0.1, 'f2'),
    #             #                             (freq[3] - 0.1, freq[3] + 0.1, 'f3'),
    #             #                             (freq[4] - 0.1, freq[4] + 0.1, 'f4')],
    #             #                      kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})
    #
    #             bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
    #                                 bandpass=False,
    #                                 bands=[(freq[0] - 0.1, freq[0] + 0.1, 'f0'),
    #                                        (freq[1] - 0.1, freq[1] + 0.1, 'f1'),
    #                                        (freq[2] - 0.1, freq[2] + 0.1, 'f2'),
    #                                        (freq[3] - 0.1, freq[3] + 0.1, 'f3'),
    #                                        (freq[4] - 0.1, freq[4] + 0.1, 'f4')],
    #                                 kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})
    #
    #             # Extract the absolute power from the relative powers output above
    #             bands = ['f0', 'f1', 'f2', 'f3', 'f4']
    #             bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
    #             bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
    #             power = bp_abs['Sum'].values  # Extract value of interest
    #
    #             # Now have power for each subject, insert it into the correct condition array
    #             if cond_name == 'median':
    #                 pow_med_pca[subject - 1, :] = power
    #             elif cond_name == 'tibial':
    #                 pow_tib_pca[subject - 1, :] = power
    #
    #     # Save to file
    #     savepow.pow_med = pow_med_pca
    #     savepow.pow_tib = pow_tib_pca
    #     dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/ecg_rm_py/inps_yasa.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(savepow, keyword))
    #
    # ##########################################################################
    # # Calculate Power for PCA PCHIP Data
    # ##########################################################################
    # if calc_PCA_pchip:
    #     class save_pow():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     savepow = save_pow()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     pow_med_pca = np.zeros((len(subjects), 39))
    #     pow_tib_pca = np.zeros((len(subjects), 39))
    #
    #     for subject in subjects:
    #         for cond_name in cond_names:
    #             if cond_name == 'tibial':
    #                 trigger_name = 'qrs'
    #                 nerve = 2
    #             elif cond_name == 'median':
    #                 trigger_name = 'qrs'
    #                 nerve = 1
    #
    #             subject_id = f'sub-{str(subject).zfill(3)}'
    #
    #             # Want the RMS of the data
    #             # Load epochs resulting from PCA - the raw data in this folder has not been rereferenced
    #             input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
    #             fname = f"data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif"
    #             raw = mne.io.read_raw_fif(input_path+fname, preload=True)
    #
    #             freq = get_harmonics(raw, trigger_name, sampling_rate)
    #
    #             mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
    #
    #             raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
    #                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    #
    #             raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    #
    #             # Now we have the raw data in the filtered form
    #             # We have the fundamental frequency and harmonics for this subject
    #             # Compute power at the frequencies
    #             freq = np.around(freq, decimals=1)
    #
    #             data = raw.pick_channels(esg_chans).reorder_channels(esg_chans)._data * 1e6
    #             # PSD will have units uV^2 now
    #             # Need a frequency resolution of 0.1Hz - win set to 10 seconds
    #             # This outputs a dataframe
    #             # bp1 = yasa.bandpower(raw, win_sec=10, relative=True,
    #             #                      bandpass=False,
    #             #                      bands=[(freq[0] - 0.1, freq[0] + 0.1, 'f0'),
    #             #                             (freq[1] - 0.1, freq[1] + 0.1, 'f1'),
    #             #                             (freq[2] - 0.1, freq[2] + 0.1, 'f2'),
    #             #                             (freq[3] - 0.1, freq[3] + 0.1, 'f3'),
    #             #                             (freq[4] - 0.1, freq[4] + 0.1, 'f4')],
    #             #                      kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})
    #
    #             bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
    #                                 bandpass=False,
    #                                 bands=[(freq[0] - 0.1, freq[0] + 0.1, 'f0'),
    #                                        (freq[1] - 0.1, freq[1] + 0.1, 'f1'),
    #                                        (freq[2] - 0.1, freq[2] + 0.1, 'f2'),
    #                                        (freq[3] - 0.1, freq[3] + 0.1, 'f3'),
    #                                        (freq[4] - 0.1, freq[4] + 0.1, 'f4')],
    #                                 kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})
    #
    #             # Extract the absolute power from the relative powers output above
    #             bands = ['f0', 'f1', 'f2', 'f3', 'f4']
    #             bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
    #             bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
    #             power = bp_abs['Sum'].values  # Extract value of interest
    #
    #             # Now have power for each subject, insert it into the correct condition array
    #             if cond_name == 'median':
    #                 pow_med_pca[subject - 1, :] = power
    #             elif cond_name == 'tibial':
    #                 pow_tib_pca[subject - 1, :] = power
    #
    #     # Save to file
    #     savepow.pow_med = pow_med_pca
    #     savepow.pow_tib = pow_tib_pca
    #     dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/ecg_rm_py/inps_yasa_pchip.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(savepow, keyword))
    #
    # ##########################################################################
    # # Calculate Power for PCA Tukey Data
    # ##########################################################################
    # if calc_PCA_tukey:
    #     class save_pow():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     savepow = save_pow()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     pow_med_pca = np.zeros((len(subjects), 39))
    #     pow_tib_pca = np.zeros((len(subjects), 39))
    #
    #     for subject in subjects:
    #         for cond_name in cond_names:
    #             if cond_name == 'tibial':
    #                 trigger_name = 'qrs'
    #                 nerve = 2
    #             elif cond_name == 'median':
    #                 trigger_name = 'qrs'
    #                 nerve = 1
    #
    #             subject_id = f'sub-{str(subject).zfill(3)}'
    #
    #             # Want the RMS of the data
    #             # Load epochs resulting from PCA - the raw data in this folder has not been rereferenced
    #             input_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/" + subject_id + "/esg/prepro/"
    #             fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
    #             raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    #
    #             freq = get_harmonics(raw, trigger_name, sampling_rate)
    #
    #             mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
    #
    #             raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
    #                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    #
    #             raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    #
    #             # Now we have the raw data in the filtered form
    #             # We have the fundamental frequency and harmonics for this subject
    #             # Compute power at the frequencies
    #             freq = np.around(freq, decimals=1)
    #
    #             data = raw.pick_channels(esg_chans).reorder_channels(esg_chans)._data * 1e6
    #             # PSD will have units uV^2 now
    #             bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
    #                                 bandpass=False,
    #                                 bands=[(freq[0] - 0.1, freq[0] + 0.1, 'f0'),
    #                                        (freq[1] - 0.1, freq[1] + 0.1, 'f1'),
    #                                        (freq[2] - 0.1, freq[2] + 0.1, 'f2'),
    #                                        (freq[3] - 0.1, freq[3] + 0.1, 'f3'),
    #                                        (freq[4] - 0.1, freq[4] + 0.1, 'f4')],
    #                                 kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})
    #
    #             # Extract the absolute power from the relative powers output above
    #             bands = ['f0', 'f1', 'f2', 'f3', 'f4']
    #             bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
    #             bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
    #             power = bp_abs['Sum'].values  # Extract value of interest
    #
    #             # Now have power for each subject, insert it into the correct condition array
    #             if cond_name == 'median':
    #                 pow_med_pca[subject - 1, :] = power
    #             elif cond_name == 'tibial':
    #                 pow_tib_pca[subject - 1, :] = power
    #
    #     # Save to file
    #     savepow.pow_med = pow_med_pca
    #     savepow.pow_tib = pow_tib_pca
    #     dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/inps_yasa.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(savepow, keyword))
    #
    # ##########################################################################
    # # Calculate Power for PCA Tukey Data
    # ##########################################################################
    # if calc_PCA_tukey_pchip:
    #     class save_pow():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     savepow = save_pow()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     pow_med_pca = np.zeros((len(subjects), 39))
    #     pow_tib_pca = np.zeros((len(subjects), 39))
    #
    #     for subject in subjects:
    #         for cond_name in cond_names:
    #             if cond_name == 'tibial':
    #                 trigger_name = 'qrs'
    #                 nerve = 2
    #             elif cond_name == 'median':
    #                 trigger_name = 'qrs'
    #                 nerve = 1
    #
    #             subject_id = f'sub-{str(subject).zfill(3)}'
    #
    #             # Want the RMS of the data
    #             # Load epochs resulting from PCA - the raw data in this folder has not been rereferenced
    #             input_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/" + subject_id + "/esg/prepro/"
    #             fname = f"data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif"
    #             raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    #
    #             freq = get_harmonics(raw, trigger_name, sampling_rate)
    #
    #             mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
    #
    #             raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
    #                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    #
    #             raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    #
    #             # Now we have the raw data in the filtered form
    #             # We have the fundamental frequency and harmonics for this subject
    #             # Compute power at the frequencies
    #             freq = np.around(freq, decimals=1)
    #
    #             data = raw.pick_channels(esg_chans).reorder_channels(esg_chans)._data * 1e6
    #             # PSD will have units uV^2 now
    #             bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
    #                                 bandpass=False,
    #                                 bands=[(freq[0] - 0.1, freq[0] + 0.1, 'f0'),
    #                                        (freq[1] - 0.1, freq[1] + 0.1, 'f1'),
    #                                        (freq[2] - 0.1, freq[2] + 0.1, 'f2'),
    #                                        (freq[3] - 0.1, freq[3] + 0.1, 'f3'),
    #                                        (freq[4] - 0.1, freq[4] + 0.1, 'f4')],
    #                                 kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})
    #
    #             # Extract the absolute power from the relative powers output above
    #             bands = ['f0', 'f1', 'f2', 'f3', 'f4']
    #             bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
    #             bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
    #             power = bp_abs['Sum'].values  # Extract value of interest
    #
    #             # Now have power for each subject, insert it into the correct condition array
    #             if cond_name == 'median':
    #                 pow_med_pca[subject - 1, :] = power
    #             elif cond_name == 'tibial':
    #                 pow_tib_pca[subject - 1, :] = power
    #
    #     # Save to file
    #     savepow.pow_med = pow_med_pca
    #     savepow.pow_tib = pow_tib_pca
    #     dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/inps_yasa_pchip.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(savepow, keyword))
    #
    # ##########################################################################
    # # Calculate Power for ICA Data
    # ##########################################################################
    # if calc_ICA:
    #     class save_pow():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     savepow = save_pow()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     pow_med_ica = np.zeros((len(subjects), 39))
    #     pow_tib_ica = np.zeros((len(subjects), 39))
    #
    #     for subject in subjects:
    #         for cond_name in cond_names:
    #             if cond_name == 'tibial':
    #                 trigger_name = 'qrs'
    #                 nerve = 2
    #             elif cond_name == 'median':
    #                 trigger_name = 'qrs'
    #                 nerve = 1
    #
    #             subject_id = f'sub-{str(subject).zfill(3)}'
    #
    #             # Want the RMS of the data
    #             # Load epochs resulting from ICA
    #             input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
    #             if choose_limited:
    #                 fname = f"clean_baseline_ica_auto_{cond_name}_lim.fif"
    #             else:
    #                 fname = f"clean_baseline_ica_auto_{cond_name}.fif"
    #             raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    #
    #             freq = get_harmonics(raw, trigger_name, sampling_rate)
    #
    #             # Now we have the raw data in the filtered form
    #             # We have the fundamental frequency and harmonics for this subject
    #             # Compute power at the frequencies
    #             # freq = np.around(freq, decimals=1)
    #             # data = raw.get_data(esg_chans) * 1e6  # Both give same answer
    #             freq = np.around(freq, decimals=1)
    #
    #             data = raw.pick_channels(esg_chans).reorder_channels(esg_chans)._data * 1e6
    #             # PSD will have units uV^2 now
    #             # Need a frequency resolution of 0.1Hz - win set to 10 seconds
    #             # This outputs a dataframe
    #             # bp1 = yasa.bandpower(raw, win_sec=10, relative=True,
    #             #                      bandpass=False,
    #             #                      bands=[(freq[0] - 0.1, freq[0] + 0.1, 'f0'),
    #             #                             (freq[1] - 0.1, freq[1] + 0.1, 'f1'),
    #             #                             (freq[2] - 0.1, freq[2] + 0.1, 'f2'),
    #             #                             (freq[3] - 0.1, freq[3] + 0.1, 'f3'),
    #             #                             (freq[4] - 0.1, freq[4] + 0.1, 'f4')],
    #             #                      kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})
    #
    #             bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
    #                                 bandpass=False,
    #                                 bands=[(freq[0] - 0.1, freq[0] + 0.1, 'f0'),
    #                                        (freq[1] - 0.1, freq[1] + 0.1, 'f1'),
    #                                        (freq[2] - 0.1, freq[2] + 0.1, 'f2'),
    #                                        (freq[3] - 0.1, freq[3] + 0.1, 'f3'),
    #                                        (freq[4] - 0.1, freq[4] + 0.1, 'f4')],
    #                                 kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})
    #
    #             # Extract the absolute power from the relative powers output above
    #             bands = ['f0', 'f1', 'f2', 'f3', 'f4']
    #             bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
    #             bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
    #             power = bp_abs['Sum'].values  # Extract value of interest
    #
    #             # Now have power for each subject, insert it into the correct condition array
    #             if cond_name == 'median':
    #                 pow_med_ica[subject - 1, :] = power
    #             elif cond_name == 'tibial':
    #                 pow_tib_ica[subject - 1, :] = power
    #
    #     # Save to file
    #     savepow.pow_med = pow_med_ica
    #     savepow.pow_tib = pow_tib_ica
    #     dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]
    #
    #     if choose_limited:
    #         fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa_lim.h5"
    #     else:
    #         fn = f"/data/pt_02569/tmp_data/baseline_ica_py/inps_yasa.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(savepow, keyword))
    #
    #
    # ##########################################################################
    # # Calculate Power for Post ICA Data
    # ##########################################################################
    # if calc_post_ICA:
    #     class save_pow():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     savepow = save_pow()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     pow_med_post_ica = np.zeros((len(subjects), 39))
    #     pow_tib_post_ica = np.zeros((len(subjects), 39))
    #
    #     for subject in subjects:
    #         for cond_name in cond_names:
    #             if cond_name == 'tibial':
    #                 trigger_name = 'qrs'
    #                 nerve = 2
    #             elif cond_name == 'median':
    #                 trigger_name = 'qrs'
    #                 nerve = 1
    #
    #             subject_id = f'sub-{str(subject).zfill(3)}'
    #
    #             # Want the RMS of the data
    #             # Load epochs resulting from ICA the data
    #             input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
    #             fname = f"clean_ica_auto_{cond_name}.fif"
    #             raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    #
    #             freq = get_harmonics(raw, trigger_name, sampling_rate)
    #
    #             # Now we have the raw data in the filtered form
    #             # We have the fundamental frequency and harmonics for this subject
    #             # Compute power at the frequencies
    #             freq = np.around(freq, decimals=1)
    #             data = raw.pick_channels(esg_chans).reorder_channels(esg_chans)._data * 1e6
    #             # PSD will have units uV^2 now
    #             # Need a frequency resolution of 0.1Hz - win set to 10 seconds
    #             # This outputs a dataframe
    #             bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
    #                                 bandpass=False,
    #                                 bands=[(freq[0] - 0.1, freq[0] + 0.1, 'f0'),
    #                                        (freq[1] - 0.1, freq[1] + 0.1, 'f1'),
    #                                        (freq[2] - 0.1, freq[2] + 0.1, 'f2'),
    #                                        (freq[3] - 0.1, freq[3] + 0.1, 'f3'),
    #                                        (freq[4] - 0.1, freq[4] + 0.1, 'f4')],
    #                                 kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})
    #
    #             # Extract the absolute power from the relative powers output above
    #             bands = ['f0', 'f1', 'f2', 'f3', 'f4']
    #             bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
    #             bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
    #             power = bp_abs['Sum'].values  # Extract value of interest
    #
    #             # Now have power for each subject, insert it into the correct condition array
    #             if cond_name == 'median':
    #                 pow_med_post_ica[subject - 1, :] = power
    #             elif cond_name == 'tibial':
    #                 pow_tib_post_ica[subject - 1, :] = power
    #
    #     # Save to file
    #     savepow.pow_med = pow_med_post_ica
    #     savepow.pow_tib = pow_tib_post_ica
    #     dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/ica_py/inps_yasa.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(savepow, keyword))
    #
    # ##########################################################################
    # # Calculate Power for SSP Data
    # ##########################################################################
    # if calc_SSP:
    #     class save_pow():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     savepow = save_pow()
    #
    #     for n in np.arange(5, 21):
    #         # Matrix of dimensions no.subjects x no. channel
    #         pow_med_ssp = np.zeros((len(subjects), 39))
    #         pow_tib_ssp = np.zeros((len(subjects), 39))
    #
    #         for subject in subjects:
    #             for cond_name in cond_names:
    #                 if cond_name == 'tibial':
    #                     trigger_name = 'qrs'
    #                     nerve = 2
    #                 elif cond_name == 'median':
    #                     trigger_name = 'qrs'
    #                     nerve = 1
    #
    #                 subject_id = f'sub-{str(subject).zfill(3)}'
    #
    #                 # Want the RMS of the data
    #                 # Load epochs resulting from SSP the data
    #                 input_path = "/data/p_02569/SSP/" + subject_id
    #                 savename = input_path + "/" + str(n) + " projections/"
    #                 raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")
    #
    #                 freq = get_harmonics(raw, trigger_name, sampling_rate)
    #
    #                 # Now we have the raw data in the filtered form
    #                 # We have the fundamental frequency and harmonics for this subject
    #                 # Compute power at the frequencies
    #                 freq = np.around(freq, decimals=1)
    #                 data = raw.get_data(esg_chans) * 1e6
    #
    #                 # PSD will have units uV^2 now
    #                 # Need a frequency resolution of 0.1Hz - win set to 10 seconds
    #                 # This outputs a dataframe
    #                 bp = yasa.bandpower(data, sf=sampling_rate, ch_names=esg_chans, win_sec=10, relative=True,
    #                                     bandpass=False,
    #                                     bands=[(freq[0] - 0.1, freq[0] + 0.1, 'f0'),
    #                                            (freq[1] - 0.1, freq[1] + 0.1, 'f1'),
    #                                            (freq[2] - 0.1, freq[2] + 0.1, 'f2'),
    #                                            (freq[3] - 0.1, freq[3] + 0.1, 'f3'),
    #                                            (freq[4] - 0.1, freq[4] + 0.1, 'f4')],
    #                                     kwargs_welch={'scaling': 'spectrum', 'average': 'median', 'window': 'hamming'})
    #
    #                 # Extract the absolute power from the relative powers output above
    #                 bands = ['f0', 'f1', 'f2', 'f3', 'f4']
    #                 bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
    #                 bp_abs['Sum'] = bp_abs.sum(axis=1)  # Get the sum across fundamental frequency and harmonics
    #                 power = bp_abs['Sum'].values  # Extract value of interest
    #
    #                 # Now have power for each subject, insert it into the correct condition array
    #                 if cond_name == 'median':
    #                     pow_med_ssp[subject - 1, :] = power
    #                 elif cond_name == 'tibial':
    #                     pow_tib_ssp[subject - 1, :] = power
    #
    #         # Save to file
    #         savepow.pow_med = pow_med_ssp
    #         savepow.pow_tib = pow_tib_ssp
    #         dataset_keywords = [a for a in dir(savepow) if not a.startswith('__')]
    #
    #         fn = f"/data/p_02569/SSP/inps_yasa_{n}.h5"
    #
    #         with h5py.File(fn, "w") as outfile:
    #             for keyword in dataset_keywords:
    #                 outfile.create_dataset(keyword, data=getattr(savepow, keyword))

