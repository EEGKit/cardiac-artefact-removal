# File to compute the inpsr using the bandpower functionality of the yasa package

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
    which_method = {'Prep': False,
                    'PCA': False,
                    'PCA PCHIP': False,
                    'PCA Tukey': False,
                    'PCA Tukey PCHIP': False,
                    'ICA': False,
                    'Post-ICA': False,
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
                for n in np.arange(1, 6):  # 5, 21
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

                            # Compute power at the fundamental frequency and harmonics + get power
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
    # Calculate INPS for each - Prepared divided by cleaned
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
            for n in np.arange(1, 21):  # 5, 21
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
            for n in np.arange(1, 21):  # 5, 21
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

