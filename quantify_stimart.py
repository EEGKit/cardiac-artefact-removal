# Stimulation artefact is interpolated from -7ms to 7ms about the stimulation point
# Want to calculate the standard deviation in this period to see effect of cleaning/filtering on this
# Calculating the standard deviation of the evoked response

import mne
import numpy as np
import h5py
from scipy.io import loadmat
from SNR_functions import evoked_from_raw

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

        # Matrix of dimensions no.subjects x no. projections
        std_med = np.zeros((len(subjects), 1))
        std_tib = np.zeros((len(subjects), 1))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the STD 7ms about stimulation point
                # Load epochs resulting from PCA OBS cleaning - the raw data in this folder has not been rereferenced
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}.fif", preload=True)

                mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                # Using the same evoked parameters used to get the evoked signal SNR
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)



                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                    chan_med.append(chan)
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr
                    chan_tib.append(chan)

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

        fn = f"/data/pt_02569/tmp_data/prepared_py/snr.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))
