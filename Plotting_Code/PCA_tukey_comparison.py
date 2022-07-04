# Just want to compare the evoked responses from SSP before and after anterior rereferncing to see difference

import mne
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw

if __name__ == '__main__':
    reduced_epochs = False
    single_subject = True
    grand_average = False

    # Testing with random subjects atm
    subjects = [1, 20]
    # subjects = np.arange(1, 2)  # (1, 37) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    if single_subject:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/PCA_tukey_comparison_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channels = ['S23', 'L1', 'S31']
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channels = ['S6', 'SC6', 'S14']

                # Load epochs resulting from SSP cleaning
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id +"/esg/prepro/"
                input_path_tuk = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/" + subject_id +"/esg/prepro/"
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(f"{input_path}{fname}")
                raw_tuk = mne.io.read_raw_fif(f"{input_path_tuk}{fname}")

                for ch in channels:
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                    evoked_tuk = evoked_from_raw(raw_tuk, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                    evoked_ch = evoked.pick_channels([ch])
                    evoked_tuk_ch = evoked_tuk.pick_channels([ch])
                    plt.figure()
                    plt.plot(evoked_ch.times, evoked_ch.data.reshape(-1), label='Original')
                    plt.plot(evoked_tuk_ch.times, evoked_tuk_ch.data.reshape(-1), label='Tukey')
                    plt.xlim([-0.025, 0.065])
                    plt.legend()
                    plt.title(f'Channel {ch}')

                    if cond_name == 'tibial':
                        plt.axvline(x=22 / 1000, linewidth=0.5, linestyle='--')
                    elif cond_name == 'median':
                        plt.axvline(x=13 / 1000, linewidth=0.5, linestyle='--')

                    if reduced_epochs:
                        plt.savefig(f"{figure_path}PCA_{ch}_{cond_name}_reduced.png")
                        print('Printing reduced')
                    else:
                        plt.savefig(f"{figure_path}PCA_{ch}_{cond_name}.png")
                        print('Printing')
                    plt.close()

            plt.show()

