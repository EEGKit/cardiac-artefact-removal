import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

if __name__ == '__main__':
    cfg_path = "/data/pt_02569/"
    cfg = loadmat(cfg_path + 'cfg.mat')
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    sampling_rate = 1000
    cond_name = 'tibial'
    subject_id = 'sub-001'

    # load dirty (prepared) ESG data
    load_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
    fname = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
    # fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
    raw = mne.io.read_raw_fif(load_path + fname, preload=True)

    # raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
    #            iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
    # raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
    plt.figure()
    fig = raw.plot_psd(0, 350)
    plt.show()
