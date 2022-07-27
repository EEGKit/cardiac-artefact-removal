##############  File to compute the residual intensity after cleaning  #################
# BCG residual intensity was quantified by calculating the root mean square (RMS) of the EEG data,
# epoched over windows of 600 ms after the ECG peaks and averaged. The percentage of BCG residual intensity
# was defined as the ratio between RMS of EEG data after and before BCG correction, respectively.
# – From the adaptive optimal basis sets paper

from scipy.io import loadmat
import numpy as np
import mne
import h5py
from SNR_functions import evoked_from_raw

if __name__ == '__main__':
    choose_limited = False  # If true, use data where only top 4 components chosen - use FALSE, see main
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

    # Want 200ms before R-peak and 400ms after R-peak
    # Baseline is the 100ms period before the artefact occurs
    iv_baseline = [-300 / 1000, -200 / 1000]
    # Want 200ms before and 400ms after the R-peak in our epoch - need baseline outside this
    iv_epoch = [-300 / 1000, 400 / 1000]
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
            class save_res():
                def __init__(self):
                    pass

            # Instantiate class
            saveres = save_res()

            if method == 'SSP':
                for n in np.arange(5, 21):
                    # Instantiate class
                    saveres = save_res()

                    # Matrix of dimensions no.subjects x no. channels
                    globals()[f"res_med_SSP_{n}"] = np.zeros((len(subjects), 39))
                    globals()[f"res_tib_SSP_{n}"] = np.zeros((len(subjects), 39))

                    for subject in subjects:
                        for cond_name in cond_names:
                            if cond_name == 'tibial':
                                trigger_name = 'qrs'
                                nerve = 2
                            elif cond_name == 'median':
                                trigger_name = 'qrs'
                                nerve = 1

                            subject_id = f'sub-{str(subject).zfill(3)}'

                            # Want the RMS of the data, load data
                            input_path = "/data/p_02569/SSP/" + subject_id
                            savename = input_path + "/" + str(n) + " projections/"
                            raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")
                            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                            # Now we have an evoked potential about the heartbeat
                            # Want to compute the RMS for each channel
                            res_chan_SSP = []
                            for ch in esg_chans:
                                # Pick a single channel
                                evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
                                data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
                                rms = np.sqrt(np.mean(data ** 2))
                                res_chan_SSP.append(rms)

                            # Now have rms for each subject, for each channel and condition
                            if cond_name == 'median':
                                globals()[f"res_med_SSP_{n}"][subject - 1, :] = res_chan_SSP
                            elif cond_name == 'tibial':
                                globals()[f"res_tib_SSP_{n}"][subject - 1, :] = res_chan_SSP

                    # Save to file
                    saveres.res_med = globals()[f"res_med_SSP_{n}"]
                    saveres.res_tib = globals()[f"res_tib_SSP_{n}"]
                    dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]

                    fn = f"/data/p_02569/SSP/res_{n}.h5"

                    with h5py.File(fn, "w") as outfile:
                        for keyword in dataset_keywords:
                            outfile.create_dataset(keyword, data=getattr(saveres, keyword))
            else:
                # Matrix of dimensions no.subjects x no. channels
                res_med = np.zeros((len(subjects), 39))
                res_tib = np.zeros((len(subjects), 39))
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

                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                        # Now we have an evoked potential about the heartbeat
                        # Want to compute the RMS for each channel
                        res_chan = []
                        for ch in esg_chans:
                            # Pick a single channel
                            evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
                            data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
                            rms = np.sqrt(np.mean(data ** 2))
                            res_chan.append(rms)

                        # Now have rms for each subject, for each channel and condition
                        if cond_name == 'median':
                            res_med[subject - 1, :] = res_chan
                        elif cond_name == 'tibial':
                            res_tib[subject - 1, :] = res_chan

                # Save to file
                saveres.res_med = res_med
                saveres.res_tib = res_tib
                dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]
                if method == 'ICA' and choose_limited:
                    fn = f"{file_path}res_lim.h5"
                elif method == 'PCA PCHIP' or method == 'PCA Tukey PCHIP':
                    fn = f"{file_path}res_pchip.h5"
                else:
                    fn = f"{file_path}res.h5"
                with h5py.File(fn, "w") as outfile:
                    for keyword in dataset_keywords:
                        outfile.create_dataset(keyword, data=getattr(saveres, keyword))

    ###################################################################################
    # Now we have all of the RMS values needed, we can calculate the residual intensity
    ###################################################################################
    input_paths = {'PCA': "/data/pt_02569/tmp_data/ecg_rm_py/",
                   'PCA PCHIP': "/data/pt_02569/tmp_data/ecg_rm_py/",
                   'PCA Tukey': "/data/pt_02569/tmp_data/ecg_rm_py_tukey/",
                   'PCA Tukey PCHIP': "/data/pt_02569/tmp_data/ecg_rm_py_tukey/",
                   'ICA': "/data/pt_02569/tmp_data/baseline_ica_py/",
                   'Post-ICA': "/data/pt_02569/tmp_data/ica_py/",
                   'SSP': "/data/p_02569/SSP/"}

    # All files are 36x39 dimensions - n_subjects x n_channels
    keywords = ['res_med', 'res_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/res.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        res_med_prep = infile[keywords[0]][()]
        res_tib_prep = infile[keywords[1]][()]

    print("\n")
    print('All Channels Residual Intensity')
    for i in np.arange(0, len(input_paths)):
        name = list(input_paths.keys())[i]
        input_path = input_paths[name]

        if name == 'SSP':
            # SSP
            for n in np.arange(5, 21):
                fn = f"/data/p_02569/SSP/res_{n}.h5"
                with h5py.File(fn, "r") as infile:
                    # Get the data
                    res_med = infile[keywords[0]][()]
                    res_tib = infile[keywords[1]][()]

                residual_med = (np.mean(res_med / res_med_prep, axis=tuple([0, 1]))) * 100
                residual_tib = (np.mean(res_tib / res_tib_prep, axis=tuple([0, 1]))) * 100

                print(f"Residual SSP Median {n}: {residual_med:.4f}")
                print(f"Residual SSP Tibial {n}: {residual_tib:.4f}")
        else:
            if name == 'ICA' and choose_limited:
                fn = f"{input_path}res_lim.h5"
            elif name == 'PCA Tukey PCHIP' or name == 'PCA PCHIP':
                fn = f"{input_path}res_pchip.h5"
            else:
                fn = f"{input_path}res.h5"
            with h5py.File(fn, "r") as infile:
                # Get the data
                res_med = infile[keywords[0]][()]
                res_tib = infile[keywords[1]][()]

            residual_med = (np.mean(res_med / res_med_prep, axis=tuple([0, 1]))) * 100
            residual_tib = (np.mean(res_tib / res_tib_prep, axis=tuple([0, 1]))) * 100

            print(f'Residual {name} Median: {residual_med:.4f}')
            print(f'Residual {name} Tibial: {residual_tib:.4f}')

    ############################################################################################
    # Now look at residual intensity for just our channels of interest
    #     if cond_name == 'tibial':
    #         channels = ['S23', 'L1', 'S31']
    #     elif cond_name == 'median':
    #         channels = ['S6', 'SC6', 'S14']
    ###########################################################################################
    median_pos = []
    tibial_pos = []
    for channel in ['S23', 'L1', 'S31']:
        tibial_pos.append(esg_chans.index(channel))
    for channel in ['S6', 'SC6', 'S14']:
        median_pos.append(esg_chans.index(channel))

    print("\n")
    print('Relevant Channels Residual Intensity')
    for i in np.arange(0, len(input_paths)):
        name = list(input_paths.keys())[i]
        input_path = input_paths[name]

        if name == 'SSP':
            # SSP
            for n in np.arange(5, 21):
                fn = f"/data/p_02569/SSP/res_{n}.h5"
                with h5py.File(fn, "r") as infile:
                    # Get the data
                    res_med = infile[keywords[0]][()]
                    res_tib = infile[keywords[1]][()]

                residual_med = (np.mean(res_med[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
                residual_tib = (np.mean(res_tib[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100

                print(f"Residual SSP Median {n}: {residual_med:.4f}")
                print(f"Residual SSP Tibial {n}: {residual_tib:.4f}")
        else:
            if name == 'ICA' and choose_limited:
                fn = f"{input_path}res_lim.h5"
            elif name == 'PCA Tukey PCHIP' or name == 'PCA PCHIP':
                fn = f"{input_path}res_pchip.h5"
            else:
                fn = f"{input_path}res.h5"
            with h5py.File(fn, "r") as infile:
                # Get the data
                res_med = infile[keywords[0]][()]
                res_tib = infile[keywords[1]][()]

            residual_med = (np.mean(res_med[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
            residual_tib = (np.mean(res_tib[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100

            print(f'Residual {name} Median: {residual_med:.4f}')
            print(f'Residual {name} Tibial: {residual_tib:.4f}')

    # Old Printing
    # calc_prepared = False
    # calc_PCA = False
    # calc_PCA_pchip = True
    # calc_PCA_tukey = False
    # calc_PCA_tukey_pchip = False
    # calc_post_ICA = False
    # calc_ICA = False
    # ########################### All Channels ###########################
    # # Prep
    # keywords = ['res_med', 'res_tib']
    # fn = f"/data/pt_02569/tmp_data/prepared_py/res.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_prep = infile[keywords[0]][()]
    #     res_tib_prep = infile[keywords[1]][()]
    #
    # # PCA
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py/res.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_pca = infile[keywords[0]][()]
    #     res_tib_pca = infile[keywords[1]][()]
    #
    # # Axis = 0 gives shape (39, ) - one value for each channel
    # # Axis = 1 gives shape (36, ) - one value for each subject
    # # residual_med_pca = (np.mean(res_med_pca, axis=tuple([0, 1]))/np.mean(res_med_prep, axis=tuple([0, 1])))*100
    # # residual_tib_pca = (np.mean(res_tib_pca, axis=tuple([0, 1]))/np.mean(res_tib_prep, axis=tuple([0, 1])))*100
    #
    # # Changing to mean of means after ratio is already calculated
    # residual_med_pca = (np.mean(res_med_pca / res_med_prep, axis=tuple([0, 1]))) * 100
    # residual_tib_pca = (np.mean(res_tib_pca / res_tib_prep, axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual PCA Medial: {residual_med_pca:.4f}%")
    # print(f"Residual PCA Tibial: {residual_tib_pca:.4f}%")
    #
    # # PCA PCHIP
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py/res_pchip.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_pca = infile[keywords[0]][()]
    #     res_tib_pca = infile[keywords[1]][()]
    #
    # # Changing to mean of means after ratio is already calculated
    # residual_med_pca = (np.mean(res_med_pca / res_med_prep, axis=tuple([0, 1]))) * 100
    # residual_tib_pca = (np.mean(res_tib_pca / res_tib_prep, axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual PCA PCHIP Medial: {residual_med_pca:.4f}%")
    # print(f"Residual PCA PCHIP Tibial: {residual_tib_pca:.4f}%")
    #
    # # PCA Tukey
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/res.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_pca = infile[keywords[0]][()]
    #     res_tib_pca = infile[keywords[1]][()]
    #
    # # Changing to mean of means after ratio is already calculated
    # residual_med_pca = (np.mean(res_med_pca / res_med_prep, axis=tuple([0, 1]))) * 100
    # residual_tib_pca = (np.mean(res_tib_pca / res_tib_prep, axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual PCA Tukey Medial: {residual_med_pca:.4f}%")
    # print(f"Residual PCA Tukey Tibial: {residual_tib_pca:.4f}%")
    #
    # # PCA Tukey pchip
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/res_pchip.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_pca = infile[keywords[0]][()]
    #     res_tib_pca = infile[keywords[1]][()]
    #
    # # Changing to mean of means after ratio is already calculated
    # residual_med_pca = (np.mean(res_med_pca / res_med_prep, axis=tuple([0, 1]))) * 100
    # residual_tib_pca = (np.mean(res_tib_pca / res_tib_prep, axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual PCA Tukey PCHIP Medial: {residual_med_pca:.4f}%")
    # print(f"Residual PCA Tukey PCHIP Tibial: {residual_tib_pca:.4f}%")
    #
    # # ICA
    # if choose_limited:
    #     fn = f"/data/pt_02569/tmp_data/baseline_ica_py/res_lim.h5"
    # else:
    #     fn = f"/data/pt_02569/tmp_data/baseline_ica_py/res.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_ica = infile[keywords[0]][()]
    #     res_tib_ica = infile[keywords[1]][()]
    #
    # # Axis = 0 gives shape (39, ) - one value for each channel
    # # Axis = 1 gives shape (36, ) - one value for each subject
    # # residual_med_ica = (np.mean(res_med_ica, axis=tuple([0, 1])) / np.mean(res_med_prep, axis=tuple([0, 1]))) * 100
    # # residual_tib_ica = (np.mean(res_tib_ica, axis=tuple([0, 1])) / np.mean(res_tib_prep, axis=tuple([0, 1]))) * 100
    #
    # residual_med_ica = (np.mean(res_med_ica / res_med_prep, axis=tuple([0, 1]))) * 100
    # residual_tib_ica = (np.mean(res_tib_ica / res_tib_prep, axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual ICA Medial: {residual_med_ica:.4f}%")
    # print(f"Residual ICA Tibial: {residual_tib_ica:.4f}%")
    #
    # # Post-ICA
    # fn = f"/data/pt_02569/tmp_data/ica_py/res.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_post_ica = infile[keywords[0]][()]
    #     res_tib_post_ica = infile[keywords[1]][()]
    #
    # # Axis = 0 gives shape (39, ) - one value for each channel
    # # Axis = 1 gives shape (36, ) - one value for each subject
    # # residual_med_post_ica = (np.mean(res_med_post_ica, axis=tuple([0, 1])) / np.mean(res_med_prep, axis=tuple([0, 1]))) * 100
    # # residual_tib_post_ica = (np.mean(res_tib_post_ica, axis=tuple([0, 1])) / np.mean(res_tib_prep, axis=tuple([0, 1]))) * 100
    #
    # residual_med_post_ica = (np.mean(res_med_post_ica / res_med_prep, axis=tuple([0, 1]))) * 100
    # residual_tib_post_ica = (np.mean(res_tib_post_ica / res_tib_prep, axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual Post-ICA Medial: {residual_med_post_ica:.4f}%")
    # print(f"Residual Post-ICA Tibial: {residual_tib_post_ica:.4f}%")
    #
    # # SSP
    # for n in np.arange(5, 21):
    #     fn = f"/data/p_02569/SSP/res_{n}.h5"
    #     with h5py.File(fn, "r") as infile:
    #         # Get the data
    #         res_med_ssp = infile[keywords[0]][()]
    #         res_tib_ssp = infile[keywords[1]][()]
    #
    #     # Axis = 0 gives shape (39, ) - one value for each channel
    #     # Axis = 1 gives shape (36, ) - one value for each subject
    #     # residual_med_ssp = (np.mean(res_med_ssp, axis=tuple([0, 1])) / np.mean(res_med_prep, axis=tuple([0, 1]))) * 100
    #     # residual_tib_ssp = (np.mean(res_tib_ssp, axis=tuple([0, 1])) / np.mean(res_tib_prep, axis=tuple([0, 1]))) * 100
    #
    #     residual_med_ssp = (np.mean(res_med_ssp / res_med_prep, axis=tuple([0, 1]))) * 100
    #     residual_tib_ssp = (np.mean(res_tib_ssp / res_tib_prep, axis=tuple([0, 1]))) * 100
    #
    #     print(f"Residual SSP Medial {n}: {residual_med_ssp:.4f}%")
    #     print(f"Residual SSP Tibial {n}: {residual_tib_ssp:.4f}%")
    #
    #
    # ############################################################################################
    # # Now look at residual intensity for just our channels of interest
    # #     if cond_name == 'tibial':
    # #         channels = ['S23', 'L1', 'S31']
    # #     elif cond_name == 'median':
    # #         channels = ['S6', 'SC6', 'S14']
    # ###########################################################################################
    # print('\n')
    # median_pos = []
    # tibial_pos = []
    # for channel in ['S23', 'L1', 'S31']:
    #     tibial_pos.append(esg_chans.index(channel))
    # for channel in ['S6', 'SC6', 'S14']:
    #     median_pos.append(esg_chans.index(channel))
    #
    # # All files are 36x39 dimensions - n_subjects x n_channels
    # keywords = ['res_med', 'res_tib']
    # fn = f"/data/pt_02569/tmp_data/prepared_py/res.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_prep = infile[keywords[0]][()]
    #     res_tib_prep = infile[keywords[1]][()]
    #
    # # PCA
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py/res.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_pca = infile[keywords[0]][()]
    #     res_tib_pca = infile[keywords[1]][()]
    #
    # # Axis = 0 gives shape (39, ) - one value for each channel
    # # Axis = 1 gives shape (36, ) - one value for each subject
    # # residual_med_pca = (np.mean(res_med_pca[:, median_pos], axis=tuple([0, 1])) / np.mean(res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    # # residual_tib_pca = (np.mean(res_tib_pca[:, tibial_pos], axis=tuple([0, 1])) / np.mean(res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100
    #
    # residual_med_pca = (np.mean(res_med_pca[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    # residual_tib_pca = (np.mean(res_tib_pca[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual PCA Medial: {residual_med_pca:.4f}%")
    # print(f"Residual PCA Tibial: {residual_tib_pca:.4f}%")
    #
    # # PCA PCHIP
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py/res_pchip.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_pca = infile[keywords[0]][()]
    #     res_tib_pca = infile[keywords[1]][()]
    #
    # residual_med_pca = (np.mean(res_med_pca[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    # residual_tib_pca = (np.mean(res_tib_pca[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual PCA PCHIP Medial: {residual_med_pca:.4f}%")
    # print(f"Residual PCA PCHIP Tibial: {residual_tib_pca:.4f}%")
    #
    # # PCA Tukey
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/res.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_pca = infile[keywords[0]][()]
    #     res_tib_pca = infile[keywords[1]][()]
    #
    # residual_med_pca = (np.mean(res_med_pca[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    # residual_tib_pca = (np.mean(res_tib_pca[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual PCA Tukey Medial: {residual_med_pca:.4f}%")
    # print(f"Residual PCA Tukey Tibial: {residual_tib_pca:.4f}%")
    #
    # # PCA Tukey PCHIP
    # fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/res_pchip.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_pca = infile[keywords[0]][()]
    #     res_tib_pca = infile[keywords[1]][()]
    #
    # residual_med_pca = (np.mean(res_med_pca[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    # residual_tib_pca = (np.mean(res_tib_pca[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual PCA Tukey PCHIP Medial: {residual_med_pca:.4f}%")
    # print(f"Residual PCA Tukey PCHIP Tibial: {residual_tib_pca:.4f}%")
    #
    # # ICA
    # if choose_limited:
    #     fn = f"/data/pt_02569/tmp_data/baseline_ica_py/res_lim.h5"
    # else:
    #     fn = f"/data/pt_02569/tmp_data/baseline_ica_py/res.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_ica = infile[keywords[0]][()]
    #     res_tib_ica = infile[keywords[1]][()]
    #
    # # Axis = 0 gives shape (39, ) - one value for each channel
    # # Axis = 1 gives shape (36, ) - one value for each subject
    # # residual_med_ica = (np.mean(res_med_ica[:, median_pos], axis=tuple([0, 1])) / np.mean(res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    # # residual_tib_ica = (np.mean(res_tib_ica[:, tibial_pos], axis=tuple([0, 1])) / np.mean(res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100
    #
    # residual_med_ica = (np.mean(res_med_ica[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    # residual_tib_ica = (np.mean(res_tib_ica[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual ICA Medial: {residual_med_ica:.4f}%")
    # print(f"Residual ICA Tibial: {residual_tib_ica:.4f}%")
    #
    # # Post-ICA
    # fn = f"/data/pt_02569/tmp_data/ica_py/res.h5"
    # with h5py.File(fn, "r") as infile:
    #     # Get the data
    #     res_med_post_ica = infile[keywords[0]][()]
    #     res_tib_post_ica = infile[keywords[1]][()]
    #
    # # Axis = 0 gives shape (39, ) - one value for each channel
    # # Axis = 1 gives shape (36, ) - one value for each subject
    # # residual_med_post_ica = (np.mean(res_med_post_ica[:, median_pos], axis=tuple([0, 1])) / np.mean(res_med_prep[:, median_pos],
    # #                                                                                  axis=tuple([0, 1]))) * 100
    # # residual_tib_post_ica = (np.mean(res_tib_post_ica[:, tibial_pos], axis=tuple([0, 1])) / np.mean(res_tib_prep[:, tibial_pos],
    # #                                                                                  axis=tuple([0, 1]))) * 100
    #
    # residual_med_post_ica = (np.mean(res_med_post_ica[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    # residual_tib_post_ica = (np.mean(res_tib_post_ica[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100
    #
    # print(f"Residual Post-ICA Medial: {residual_med_post_ica:.4f}%")
    # print(f"Residual Post-ICA Tibial: {residual_tib_post_ica:.4f}%")
    #
    # # SSP
    # for n in np.arange(5, 21):
    #     fn = f"/data/p_02569/SSP/res_{n}.h5"
    #     with h5py.File(fn, "r") as infile:
    #         # Get the data
    #         res_med_ssp = infile[keywords[0]][()]
    #         res_tib_ssp = infile[keywords[1]][()]
    #
    #     # Axis = 0 gives shape (39, ) - one value for each channel
    #     # Axis = 1 gives shape (36, ) - one value for each subject
    #     # residual_med_ssp = (np.mean(res_med_ssp[:, median_pos], axis=tuple([0, 1])) / np.mean(res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    #     # residual_tib_ssp = (np.mean(res_tib_ssp[:, tibial_pos], axis=tuple([0, 1])) / np.mean(res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100
    #
    #     residual_med_ssp = (np.mean(res_med_ssp[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
    #     residual_tib_ssp = (np.mean(res_tib_ssp[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100
    #
    #     print(f"Residual SSP Medial {n}: {residual_med_ssp:.4f}%")
    #     print(f"Residual SSP Tibial {n}: {residual_tib_ssp:.4f}%")


    ####################### Old Code #######################
    # ##########################################################################
    # # Calculate Residuals for Prepared Data
    # ##########################################################################
    # if calc_prepared:
    #     class save_res():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     saveres = save_res()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     res_med_prep = np.zeros((len(subjects), 39))
    #     res_tib_prep = np.zeros((len(subjects), 39))
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
    #             raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif", preload=True)
    #
    #             mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
    #
    #             raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
    #                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    #
    #             raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    #
    #             evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
    #             # evoked.plot()
    #
    #             # Now we have an evoked potential about the heartbeat
    #             # Want to compute the RMS for each channel
    #             res_chan_prep = []
    #             for ch in esg_chans:
    #                 # Pick a single channel
    #                 evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
    #                 data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
    #                 rms = np.sqrt(np.mean(data ** 2))
    #                 res_chan_prep.append(rms)
    #
    #             # Now have rms for each subject, for each channel and condition
    #             if cond_name == 'median':
    #                 res_med_prep[subject - 1, :] = res_chan_prep
    #             elif cond_name == 'tibial':
    #                 res_tib_prep[subject - 1, :] = res_chan_prep
    #
    #     # print(np.shape(res_med_prep))
    #     # print(res_med_prep)
    #     # print(np.shape(res_tib_prep))
    #     # print(res_tib_prep)
    #
    #     # Save to file
    #     saveres.res_med = res_med_prep
    #     saveres.res_tib = res_tib_prep
    #     dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/prepared_py/res.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(saveres, keyword))
    #
    # ##########################################################################
    # # Calculate Residuals for PCA_OBS cleaned data
    # ##########################################################################
    # if calc_PCA:
    #     class save_res():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     saveres = save_res()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     res_med_pca = np.zeros((len(subjects), 39))
    #     res_tib_pca = np.zeros((len(subjects), 39))
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
    #             # Load epochs resulting from ecg_rm_py - the raw data in this folder has not been rereferenced
    #             input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
    #             fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
    #             raw = mne.io.read_raw_fif(input_path+fname, preload=True)
    #
    #             # input_path = "/data/pt_02569/tmp_data/ecg_rm/" + subject_id + "/esg/prepro/"
    #             # fname = f"cnt_clean_ecg_spinal_{cond_name}.set"
    #             # raw = mne.io.read_raw_eeglab(input_path + fname, preload=True)
    #
    #             mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
    #
    #             raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
    #                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    #
    #             raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    #
    #             evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
    #
    #             # Now we have an evoked potential about the heartbeat
    #             # Want to compute the RMS for each channel
    #             res_chan_pca = []
    #             for ch in esg_chans:
    #                 # Pick a single channel
    #                 evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
    #                 data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
    #                 rms = np.sqrt(np.mean(data ** 2))
    #                 res_chan_pca.append(rms)
    #
    #             # Now have rms for each subject, for each channel and condition
    #             if cond_name == 'median':
    #                 res_med_pca[subject - 1, :] = res_chan_pca
    #             elif cond_name == 'tibial':
    #                 res_tib_pca[subject - 1, :] = res_chan_pca
    #
    #     # print(np.shape(res_med_pca))
    #     # print(res_med_pca)
    #     # print(np.shape(res_tib_pca))
    #     # print(res_tib_pca)
    #
    #     # Save to file
    #     saveres.res_med = res_med_pca
    #     saveres.res_tib = res_tib_pca
    #     dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/ecg_rm_py/res.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(saveres, keyword))
    #
    # ##########################################################################
    # # Calculate Residuals for PCA_OBS PCHIP cleaned data
    # ##########################################################################
    # if calc_PCA_pchip:
    #     class save_res():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     saveres = save_res()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     res_med_pca = np.zeros((len(subjects), 39))
    #     res_tib_pca = np.zeros((len(subjects), 39))
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
    #             # Load epochs resulting from ecg_rm_py - the raw data in this folder has not been rereferenced
    #             input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
    #             fname = f"data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif"
    #             raw = mne.io.read_raw_fif(input_path+fname, preload=True)
    #
    #             # input_path = "/data/pt_02569/tmp_data/ecg_rm/" + subject_id + "/esg/prepro/"
    #             # fname = f"cnt_clean_ecg_spinal_{cond_name}.set"
    #             # raw = mne.io.read_raw_eeglab(input_path + fname, preload=True)
    #
    #             mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
    #
    #             raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
    #                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    #
    #             raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    #
    #             evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
    #
    #             # Now we have an evoked potential about the heartbeat
    #             # Want to compute the RMS for each channel
    #             res_chan_pca = []
    #             for ch in esg_chans:
    #                 # Pick a single channel
    #                 evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
    #                 data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
    #                 rms = np.sqrt(np.mean(data ** 2))
    #                 res_chan_pca.append(rms)
    #
    #             # Now have rms for each subject, for each channel and condition
    #             if cond_name == 'median':
    #                 res_med_pca[subject - 1, :] = res_chan_pca
    #             elif cond_name == 'tibial':
    #                 res_tib_pca[subject - 1, :] = res_chan_pca
    #
    #     # print(np.shape(res_med_pca))
    #     # print(res_med_pca)
    #     # print(np.shape(res_tib_pca))
    #     # print(res_tib_pca)
    #
    #     # Save to file
    #     saveres.res_med = res_med_pca
    #     saveres.res_tib = res_tib_pca
    #     dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/ecg_rm_py/res_pchip.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(saveres, keyword))
    #
    # ##########################################################################
    # # Calculate Residuals for PCA_OBS Tukey cleaned data
    # ##########################################################################
    # if calc_PCA_tukey:
    #     class save_res():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     saveres = save_res()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     res_med_pca = np.zeros((len(subjects), 39))
    #     res_tib_pca = np.zeros((len(subjects), 39))
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
    #             # Load epochs resulting from ecg_rm_py - the raw data in this folder has not been rereferenced
    #             input_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/" + subject_id + "/esg/prepro/"
    #             fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
    #             raw = mne.io.read_raw_fif(input_path+fname, preload=True)
    #
    #             mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
    #
    #             raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
    #                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    #
    #             raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    #
    #             evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
    #
    #             # Now we have an evoked potential about the heartbeat
    #             # Want to compute the RMS for each channel
    #             res_chan_pca = []
    #             for ch in esg_chans:
    #                 # Pick a single channel
    #                 evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
    #                 data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
    #                 rms = np.sqrt(np.mean(data ** 2))
    #                 res_chan_pca.append(rms)
    #
    #             # Now have rms for each subject, for each channel and condition
    #             if cond_name == 'median':
    #                 res_med_pca[subject - 1, :] = res_chan_pca
    #             elif cond_name == 'tibial':
    #                 res_tib_pca[subject - 1, :] = res_chan_pca
    #
    #     # print(np.shape(res_med_pca))
    #     # print(res_med_pca)
    #     # print(np.shape(res_tib_pca))
    #     # print(res_tib_pca)
    #
    #     # Save to file
    #     saveres.res_med = res_med_pca
    #     saveres.res_tib = res_tib_pca
    #     dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/res.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(saveres, keyword))
    #
    # ##########################################################################
    # # Calculate Residuals for PCA_OBS Tukey PCHIP cleaned data
    # ##########################################################################
    # if calc_PCA_tukey_pchip:
    #     class save_res():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     saveres = save_res()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     res_med_pca = np.zeros((len(subjects), 39))
    #     res_tib_pca = np.zeros((len(subjects), 39))
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
    #             # Load epochs resulting from ecg_rm_py - the raw data in this folder has not been rereferenced
    #             input_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/" + subject_id + "/esg/prepro/"
    #             fname = f"data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif"
    #             raw = mne.io.read_raw_fif(input_path+fname, preload=True)
    #
    #             mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
    #
    #             raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
    #                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    #
    #             raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    #
    #             evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
    #
    #             # Now we have an evoked potential about the heartbeat
    #             # Want to compute the RMS for each channel
    #             res_chan_pca = []
    #             for ch in esg_chans:
    #                 # Pick a single channel
    #                 evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
    #                 data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
    #                 rms = np.sqrt(np.mean(data ** 2))
    #                 res_chan_pca.append(rms)
    #
    #             # Now have rms for each subject, for each channel and condition
    #             if cond_name == 'median':
    #                 res_med_pca[subject - 1, :] = res_chan_pca
    #             elif cond_name == 'tibial':
    #                 res_tib_pca[subject - 1, :] = res_chan_pca
    #
    #     # print(np.shape(res_med_pca))
    #     # print(res_med_pca)
    #     # print(np.shape(res_tib_pca))
    #     # print(res_tib_pca)
    #
    #     # Save to file
    #     saveres.res_med = res_med_pca
    #     saveres.res_tib = res_tib_pca
    #     dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/ecg_rm_py_tukey/res_pchip.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(saveres, keyword))
    #
    # ##########################################################################
    # # Calculate Residuals for Baseline ICA corrected data
    # ##########################################################################
    # if calc_ICA:
    #     class save_res():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     saveres = save_res()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     res_med_ica = np.zeros((len(subjects), 39))
    #     res_tib_ica = np.zeros((len(subjects), 39))
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
    #             # Load epochs resulting from baseline ica - the raw data in this folder has not been rereferenced
    #             input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
    #             if choose_limited:
    #                 fname = f"clean_baseline_ica_auto_{cond_name}_lim.fif"
    #             else:
    #                 fname = f"clean_baseline_ica_auto_{cond_name}.fif"
    #             raw = mne.io.read_raw_fif(input_path+fname, preload=True)
    #
    #             evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
    #
    #             # Now we have an evoked potential about the heartbeat
    #             # Want to compute the RMS for each channel
    #             res_chan_ica = []
    #             for ch in esg_chans:
    #                 # Pick a single channel
    #                 evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
    #                 data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
    #                 rms = np.sqrt(np.mean(data ** 2))
    #                 res_chan_ica.append(rms)
    #
    #             # Now have rms for each subject, for each channel and condition
    #             if cond_name == 'median':
    #                 res_med_ica[subject - 1, :] = res_chan_ica
    #             elif cond_name == 'tibial':
    #                 res_tib_ica[subject - 1, :] = res_chan_ica
    #
    #     # print(np.shape(res_med_ica))
    #     # print(res_med_ica)
    #     # print(np.shape(res_tib_ica))
    #     # print(res_tib_ica)
    #
    #     # Save to file
    #     saveres.res_med = res_med_ica
    #     saveres.res_tib = res_tib_ica
    #     dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]
    #
    #     if choose_limited:
    #         fn = f"/data/pt_02569/tmp_data/baseline_ica_py/res_lim.h5"
    #     else:
    #         fn = f"/data/pt_02569/tmp_data/baseline_ica_py/res.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(saveres, keyword))
    #
    # ##########################################################################
    # # Calculate Residuals for PCA_OBS and ICA corrected data
    # ##########################################################################
    # if calc_post_ICA:
    #     class save_res():
    #         def __init__(self):
    #             pass
    #
    #     # Instantiate class
    #     saveres = save_res()
    #
    #     # Matrix of dimensions no.subjects x no. channels
    #     res_med_post_ica = np.zeros((len(subjects), 39))
    #     res_tib_post_ica = np.zeros((len(subjects), 39))
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
    #             # Load epochs resulting from post ica - the raw data in this folder has not been rereferenced
    #             input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
    #             fname = f"clean_ica_auto_{cond_name}.fif"
    #             raw = mne.io.read_raw_fif(input_path+fname, preload=True)
    #
    #             evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
    #
    #             # Now we have an evoked potential about the heartbeat
    #             # Want to compute the RMS for each channel
    #             res_chan_post_ica = []
    #             for ch in esg_chans:
    #                 # Pick a single channel
    #                 evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
    #                 data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
    #                 rms = np.sqrt(np.mean(data ** 2))
    #                 res_chan_post_ica.append(rms)
    #
    #             # Now have rms for each subject, for each channel and condition
    #             if cond_name == 'median':
    #                 res_med_post_ica[subject - 1, :] = res_chan_post_ica
    #             elif cond_name == 'tibial':
    #                 res_tib_post_ica[subject - 1, :] = res_chan_post_ica
    #
    #     # print(np.shape(res_med_post_ica))
    #     # print(res_med_post_ica)
    #     # print(np.shape(res_tib_post_ica))
    #     # print(res_tib_post_ica)
    #
    #     # Save to file
    #     saveres.res_med = res_med_post_ica
    #     saveres.res_tib = res_tib_post_ica
    #     dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]
    #
    #     fn = f"/data/pt_02569/tmp_data/ica_py/res.h5"
    #
    #     with h5py.File(fn, "w") as outfile:
    #         for keyword in dataset_keywords:
    #             outfile.create_dataset(keyword, data=getattr(saveres, keyword))
    #
    # ##########################################################################
    # # Calculate Residuals for SSP corrected data
    # ##########################################################################
    # if calc_SSP:
    #     class save_res():
    #         def __init__(self):
    #             pass
    #
    #     for n in np.arange(5, 21):
    #         # Instantiate class
    #         saveres = save_res()
    #
    #         # Matrix of dimensions no.subjects x no. channels
    #         globals()[f"res_med_SSP_{n}"] = np.zeros((len(subjects), 39))
    #         globals()[f"res_tib_SSP_{n}"] = np.zeros((len(subjects), 39))
    #         # res_med_SSP = np.zeros((len(subjects), 39))
    #         # res_tib_SSP = np.zeros((len(subjects), 39))
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
    #                 # Load epochs resulting from post ica - the raw data in this folder has not been rereferenced
    #                 input_path = "/data/p_02569/SSP/" + subject_id
    #                 savename = input_path + "/" + str(n) + " projections/"
    #                 raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")
    #
    #                 evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
    #
    #                 # Now we have an evoked potential about the heartbeat
    #                 # Want to compute the RMS for each channel
    #                 res_chan_SSP = []
    #                 for ch in esg_chans:
    #                     # Pick a single channel
    #                     evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
    #                     data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
    #                     rms = np.sqrt(np.mean(data ** 2))
    #                     res_chan_SSP.append(rms)
    #
    #                 # Now have rms for each subject, for each channel and condition
    #                 if cond_name == 'median':
    #                     globals()[f"res_med_SSP_{n}"][subject - 1, :] = res_chan_SSP
    #                 elif cond_name == 'tibial':
    #                     globals()[f"res_tib_SSP_{n}"][subject - 1, :] = res_chan_SSP
    #
    #         # Save to file
    #         saveres.res_med = globals()[f"res_med_SSP_{n}"]
    #         saveres.res_tib = globals()[f"res_tib_SSP_{n}"]
    #         dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]
    #
    #         fn = f"/data/p_02569/SSP/res_{n}.h5"
    #
    #         with h5py.File(fn, "w") as outfile:
    #             for keyword in dataset_keywords:
    #                 outfile.create_dataset(keyword, data=getattr(saveres, keyword))
    #     # print(np.shape(res_med_SSP_6))
    #     # print(res_med_SSP_6)
    #     # print(np.shape(res_med_SSP_6))
    #     # print(res_tib_SSP_6)

