# Script to plot evoked potential about certain expected R-peaks - just for visual inspection

from scipy.io import loadmat
import mne
import matplotlib.pyplot as plt

# Testing with random subjects atm
subjects = [20]  # 20
# subjects = np.arange(1, 2)  # (1, 37) # 1 through 36 to access subject data
cond_names = ['tibial']  # ['median']
sampling_rate = 1000

cfg_path = "/data/pt_02569/"  # Contains important info about experiment
cfg = loadmat(cfg_path + 'cfg.mat')
iv_epoch = cfg['iv_epoch'][0] / 1000
iv_baseline = cfg['iv_baseline'][0] / 1000
notch_freq = cfg['notch_freq'][0]
esg_bp_freq = cfg['esg_bp_freq'][0]
prepared = True
PCA = False
PCA_tukey = True
ICA = False
SSP = False

for subject in subjects:
    subject_id = f'sub-{str(subject).zfill(3)}'

    for cond_name in cond_names:
        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channels = ['S23', 'L1', 'S31']
            # channels = ['L1']

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channels = ['S6', 'SC6', 'S14']
            # channels = ['SC6']

        if prepared:
            # Load epochs resulting from PCA OBS cleaning - the raw data in this folder has not been rereferenced
            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
            raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr1000_{cond_name}_withqrs.fif", preload=True)
            # add reference channel to data
            mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

            raw.pick_channels(channels)
            # raw.plot(n_channels=5)
            raw.plot(duration=4, start=518.75, clipping=6, scalings=80e-5)  # 40e-5 looks good for median
            # raw.plot(duration=2, start=784, clipping=6, scalings=80e-5)
            # raw.plot(duration=5, start=500)

        if PCA:
            # Some editing here to avoid plotting fit_end and fit_start
            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
            fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
            raw.pick_channels(channels)
            # raw.plot()
            # events = mne.pick_events(events, include=[1, 4])
            # raw.plot(duration=1.5, start=1164, clipping=6, scalings=30e-6)
            # raw.plot(duration=4, start=518.75, clipping=6, scalings=60e-6)
            raw.plot(duration=1, start=784, clipping=6, scalings=20e-6)
            plt.show()

        if PCA_tukey:
            # Some editing here to avoid plotting fit_end and fit_start
            input_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/" + subject_id + "/esg/prepro/"
            fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place
            raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

            raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
            raw.pick_channels(channels)
            # raw.plot()
            # events = mne.pick_events(events, include=[1, 4])
            # raw.plot(duration=1.5, start=1164, clipping=6, scalings=30e-6)
            # raw.plot(duration=4, start=518.75, clipping=6, scalings=60e-6)
            raw.plot(duration=1, start=784, clipping=6, scalings=30e-6)
            plt.show()

        if ICA:
            # Some editing here to avoid plotting fit_end and fit_start
            input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
            fname = f"clean_baseline_ica_auto_{cond_name}.fif"
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            raw.pick_channels(channels)
            # raw.plot()
            # events = mne.pick_events(events, include=[1, 4])
            # raw.plot(duration=4, start=518.75, clipping=6, scalings=60e-6)
            raw.plot(duration=4, start=518.75, clipping=6, scalings=20e-6)

        if SSP:
            input_path = "/data/p_02569/SSP/" + subject_id
            savename = input_path + "/5 projections/"
            raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")
            # raw.pick_channels(channels)
            # raw.plot()
            raw.plot(duration=1, start=518, n_channels=3)
            # raw.plot(duration=1, start=784, n_channels=1)

