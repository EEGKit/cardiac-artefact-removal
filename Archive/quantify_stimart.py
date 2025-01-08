# Stimulation artefact is interpolated from -7ms to 7ms about the stimulation point
# Want to calculate the standard deviation in this period to see effect of cleaning/filtering on this
# Calculating the standard deviation of the evoked response

import mne
import numpy as np
import h5py
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw

if __name__ == '__main__':
    # Set which to run
    calc_raw = False
    calc_PCA = False
    calc_ICA = False
    calc_SSP = False

    # Testing with just subject 1 at the moment
    subjects = np.arange(1, 37)  # (1, 2) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    ####################################################################################
    # Prepared data
    ####################################################################################
    if calc_raw:
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
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the STD 7ms about stimulation point
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
                raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}.fif", preload=True)

                # Using the same evoked parameters used to get the evoked signal SNR
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked.reorder_channels(esg_chans)

                # evoked.data: data array of shape(n_channels, n_times)
                start = -7/1000
                end = 7/1000
                time_idx = evoked.time_as_index([start, end])
                std = np.std(evoked.data[:, time_idx], axis=1)  # Array of shape (39, )

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    std_med[subject - 1, :] = std
                elif cond_name == 'tibial':
                    std_tib[subject - 1, :] = std

        # Save to file to compare to matlab - only for debugging
        savestd.std_med = std_med
        savestd.std_tib = std_tib
        dataset_keywords = [a for a in dir(savestd) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/prepared_py/std_stim.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savestd, keyword))

    ####################################################################################
    # PCA data
    ####################################################################################
    if calc_PCA:
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
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the STD 7ms about stimulation point
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Using the same evoked parameters used to get the evoked signal SNR
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked.reorder_channels(esg_chans)

                # evoked.data: data array of shape(n_channels, n_times)
                start = -7/1000
                end = 7/1000
                time_idx = evoked.time_as_index([start, end])
                std = np.std(evoked.data[:, time_idx], axis=1)  # Array of shape (39, )

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    std_med[subject - 1, :] = std
                elif cond_name == 'tibial':
                    std_tib[subject - 1, :] = std

        # Save to file to compare to matlab - only for debugging
        savestd.std_med = std_med
        savestd.std_tib = std_tib
        dataset_keywords = [a for a in dir(savestd) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/ecg_rm_py/std_stim.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savestd, keyword))


    ####################################################################################
    # ICA data
    ####################################################################################
    if calc_ICA:
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
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the STD 7ms about stimulation point
                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id
                fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Using the same evoked parameters used to get the evoked signal SNR
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked.reorder_channels(esg_chans)

                # evoked.data: data array of shape(n_channels, n_times)
                start = -7/1000
                end = 7/1000
                time_idx = evoked.time_as_index([start, end])
                std = np.std(evoked.data[:, time_idx], axis=1)  # Array of shape (39, )

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    std_med[subject - 1, :] = std
                elif cond_name == 'tibial':
                    std_tib[subject - 1, :] = std

        # Save to file to compare to matlab - only for debugging
        savestd.std_med = std_med
        savestd.std_tib = std_tib
        dataset_keywords = [a for a in dir(savestd) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/baseline_ica_py/std_stim.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savestd, keyword))

    ####################################################################################
    # SSP data
    ####################################################################################
    if calc_SSP:
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
                    if cond_name == 'tibial':
                        trigger_name = 'Tibial - Stimulation'
                    elif cond_name == 'median':
                        trigger_name = 'Median - Stimulation'

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Want the STD 7ms about stimulation point
                    input_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id
                    savename = input_path + "/" + str(n) + " projections/"
                    raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")

                    # Using the same evoked parameters used to get the evoked signal SNR
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                    evoked.reorder_channels(esg_chans)

                    # evoked.data: data array of shape(n_channels, n_times)
                    start = -7/1000
                    end = 7/1000
                    time_idx = evoked.time_as_index([start, end])
                    std = np.std(evoked.data[:, time_idx], axis=1)  # Array of shape (39, )

                    # Now have one snr related to each subject and condition
                    if cond_name == 'median':
                        std_med[subject - 1, :] = std
                    elif cond_name == 'tibial':
                        std_tib[subject - 1, :] = std

            # Save to file to compare to matlab - only for debugging
            savestd.std_med = std_med
            savestd.std_tib = std_tib
            dataset_keywords = [a for a in dir(savestd) if not a.startswith('__')]

            fn = f"/data/pt_02569/tmp_data/ssp_py/std_stim_{n}.h5"
            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savestd, keyword))

    #########################################################
    # Now we have all the std values, print to screen
    #########################################################
    keywords = ['std_med', 'std_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/std_stim.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_prep = infile[keywords[0]][()]
        std_tib_prep = infile[keywords[1]][()]

    print(f"STD Prep Medial: {np.mean(std_med_prep, axis=tuple([0, 1])):.4e}")
    print(f"STD Prep Tibial: {np.mean(std_tib_prep, axis=tuple([0, 1])):.4e}")

    # PCA
    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/std_stim.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_pca = infile[keywords[0]][()]
        std_tib_pca = infile[keywords[1]][()]

    print(f"STD PCA Medial: {np.mean(std_med_pca, axis=tuple([0, 1])):.4e}")
    print(f"STD PCA Tibial: {np.mean(std_tib_pca, axis=tuple([0, 1])):.4e}")

    # ICA
    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/std_stim.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_ica = infile[keywords[0]][()]
        std_tib_ica = infile[keywords[1]][()]

    print(f"STD ICA Medial: {np.mean(std_med_ica, axis=tuple([0, 1])):.4e}")
    print(f"STD ICA Tibial: {np.mean(std_tib_ica, axis=tuple([0, 1])):.4e}")

    # SSP
    for n in np.arange(5, 21):
        fn = f"/data/pt_02569/tmp_data/ssp_py/std_stim_{n}.h5"
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

    keywords = ['std_med', 'std_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/std_stim.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_prep = infile[keywords[0]][()]
        std_tib_prep = infile[keywords[1]][()]

    print(f"STD Prep Medial: {np.mean(std_med_prep[:, median_pos], axis=tuple([0, 1])):.4e}")
    print(f"STD Prep Tibial: {np.mean(std_tib_prep[:, tibial_pos], axis=tuple([0, 1])):.4e}")

    # PCA
    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/std_stim.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_pca = infile[keywords[0]][()]
        std_tib_pca = infile[keywords[1]][()]

    print(f"STD PCA Medial: {np.mean(std_med_pca[:, median_pos], axis=tuple([0, 1])):.4e}")
    print(f"STD PCA Tibial: {np.mean(std_tib_pca[:, tibial_pos], axis=tuple([0, 1])):.4e}")

    # ICA
    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/std_stim.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_ica = infile[keywords[0]][()]
        std_tib_ica = infile[keywords[1]][()]

    print(f"STD ICA Medial: {np.mean(std_med_ica[:, median_pos], axis=tuple([0, 1])):.4e}")
    print(f"STD ICA Tibial: {np.mean(std_tib_ica[:, tibial_pos], axis=tuple([0, 1])):.4e}")

    # SSP
    for n in np.arange(5, 21):
        fn = f"/data/pt_02569/tmp_data/ssp_py/std_stim_{n}.h5"
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
    keywords = ['std_med', 'std_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/std_stim.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_prep = infile[keywords[0]][()]
        std_tib_prep = infile[keywords[1]][()]

    # PCA
    fn = f"/data/pt_02569/tmp_data/ecg_rm_py/std_stim.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_pca = infile[keywords[0]][()]
        std_tib_pca = infile[keywords[1]][()]

    print(f"STD PCA Medial: {np.mean((std_med_pca-std_med_prep)/std_med_prep, axis=tuple([0, 1]))*100:.4f}%")
    print(f"STD PCA Tibial: {np.mean((std_tib_pca-std_tib_prep)/std_tib_prep, axis=tuple([0, 1]))*100:.4f}%")

    # ICA
    fn = f"/data/pt_02569/tmp_data/baseline_ica_py/std_stim.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        std_med_ica = infile[keywords[0]][()]
        std_tib_ica = infile[keywords[1]][()]

    print(f"STD ICA Medial: {np.mean((std_med_ica-std_med_prep)/std_med_prep, axis=tuple([0, 1]))*100:.4f}%")
    print(f"STD ICA Tibial: {np.mean((std_tib_ica-std_tib_prep)/std_tib_prep, axis=tuple([0, 1]))*100:.4f}%")

    # SSP
    for n in np.arange(5, 21):
        fn = f"/data/pt_02569/tmp_data/ssp_py/std_stim_{n}.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            std_med_ssp = infile[keywords[0]][()]
            std_tib_ssp = infile[keywords[1]][()]

        print(f"STD SSP Medial {n}: {np.mean((std_med_ssp-std_med_prep)/std_med_prep, axis=tuple([0, 1]))*100:.4f}%")
        print(f"STD SSP Tibial {n}: {np.mean((std_tib_ssp-std_tib_prep)/std_tib_prep, axis=tuple([0, 1]))*100:.4f}%")
