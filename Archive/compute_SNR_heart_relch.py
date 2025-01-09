# Very similar analysis to compute_SNR_heart, but instead of looking at all channels, look only at the three central
# channels deemed important in the respective tibial and median stimulation conditions and take average across these

import mne
import numpy as np
import h5py
from Metrics.SNR_functions import calculate_heart_SNR_evoked_ch, evoked_from_raw
from scipy.io import loadmat
from reref_data import rereference_data


if __name__ == '__main__':
    # Testing with just subject 1 at the moment
    subjects = np.arange(1, 37) # (1, 2) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000
    # Pattern of results is the same with the wider period - so looking specifically at the R-peak is retained
    # Baseline is the 100ms period before the artefact occurs
    # iv_baseline = [-300 / 1000, -200 / 1000]
    iv_baseline = [-150/1000, -50/1000]
    # Want 200ms before and 400ms after the R-peak
    # iv_epoch = [-300 / 1000, 400 / 1000]
    iv_epoch = [-200/1000, 200/1000]

    # Set which to run
    calc_raw_snr = True
    calc_PCA_snr = True
    calc_ICA_snr = True
    calc_SSP_snr = True
    ant_ref = False  # Use the data that has been anteriorly referenced instead
    reduced_window = False # Reduced window about expected peak
    reduced_epochs = False  # Dummy variable - always false in this script as I don't reduce epochs

    # Run SNR calc on prepared data - heart artefact NOT removed here
    ############################################## Raw SNR Calculations ########################################
    if calc_raw_snr:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 1))
        snr_tib = np.zeros((len(subjects), 1))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    nerve = 2
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    nerve = 1

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from preparation script
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
                fname = f"noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # add reference channel to data
                if ant_ref:
                    # anterior reference
                    if nerve == 1:
                        raw = rereference_data(raw, 'AC')
                    elif nerve == 2:
                        raw = rereference_data(raw, 'AL')

                cfg_path = "/data/pt_02569/"  # Contains important info about experiment
                cfg = loadmat(cfg_path + 'cfg.mat')

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, 'qrs', reduced_epochs)

                snr = calculate_heart_SNR_evoked_ch(evoked, cond_name, iv_baseline, reduced_window)

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/prepared_py/snr_heart_ch_ant_smallwin.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/prepared_py/snr_heart_ch_smallwin.h5"
        else:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/prepared_py/snr_heart_ch_ant.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/prepared_py/snr_heart_ch.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))
        # print(snr_med)
        # print(snr_tib)

    ############################################ PCA SNR Calculations ########################################
    if calc_PCA_snr:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass


        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 1))
        snr_tib = np.zeros((len(subjects), 1))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    nerve = 2
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    nerve = 1

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Data in this folder hasn't been filtered and rereferenced - do it here instead
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                # add reference channel to data
                if ant_ref:
                    # anterior reference
                    if nerve == 1:
                        raw = rereference_data(raw, 'AC')
                    elif nerve == 2:
                        raw = rereference_data(raw, 'AL')

                cfg_path = "/data/pt_02569/"  # Contains important info about experiment
                cfg = loadmat(cfg_path + 'cfg.mat')

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, 'qrs', reduced_epochs)

                snr = calculate_heart_SNR_evoked_ch(evoked, cond_name, iv_baseline, reduced_window)

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_heart_ch_ant_smallwin.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_heart_ch_smallwin.h5"
        else:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_heart_ch_ant.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_heart_ch.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

        # print(snr_med)
        # print(snr_tib)

    ######################################## ICA SNR Calculations ########################################
    if calc_ICA_snr:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 1))
        snr_tib = np.zeros((len(subjects), 1))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from preparation script
                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id
                if ant_ref:
                    fname = f"clean_baseline_ica_auto_antRef_{cond_name}.fif"
                else:
                    fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname)

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, 'qrs', reduced_epochs)

                snr = calculate_heart_SNR_evoked_ch(evoked, cond_name, iv_baseline, reduced_window)

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_heart_ch_ant_smallwin.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_heart_ch_smallwin.h5"
        else:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_heart_ch_ant.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_heart_ch.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))
        # print(snr_med)
        # print(snr_tib)

    ############################### SSP Projectors SNR #################################################
    if calc_SSP_snr:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), len(np.arange(5, 21))))
        snr_tib = np.zeros((len(subjects), len(np.arange(5, 21))))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR for each projection tried from 5 to 20
                for n in np.arange(5, 21):
                    # Load SSP projection data
                    input_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id
                    savename = input_path + "/" + str(n) + " projections/"
                    if ant_ref:
                        raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}_antRef.fif")
                    else:
                        raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")

                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, 'qrs', reduced_epochs)

                    snr = calculate_heart_SNR_evoked_ch(evoked, cond_name, iv_baseline, reduced_window)

                    # Now have one snr for relevant channel in each subject + condition
                    if cond_name == 'median':
                        snr_med[subject-1, n-5] = snr
                    elif cond_name == 'tibial':
                        snr_tib[subject - 1, n-5] = snr

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/ssp_py/snr_heart_ch_ant_smallwin.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/ssp_py/snr_heart_ch_smallwin.h5"
        else:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/ssp_py/snr_heart_ch_ant.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/ssp_py/snr_heart_ch.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))
        # print(snr_med)
        # print(snr_tib)
