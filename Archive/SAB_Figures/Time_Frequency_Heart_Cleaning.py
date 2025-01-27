# Script to create plots of the grand averages evoked responses of the heartbeat across participants for each stimulation

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from mpl_axes_aligner import align
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == '__main__':
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    # subjects = [1]
    freqs = np.arange(5., 250., 3.)
    fmin, fmax = freqs[[0, -1]]
    cond_names = ['tibial', 'median']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    iv_epoch = [-400/1000, 600/1000]
    iv_baseline = [-400/1000, -300/1000]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/TFR_Heart_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list_prep = []
        evoked_list_pca = []
        evoked_list_ica = []
        evoked_list_ssp6 = []

        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channel = ['L1']

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channel = ['SC6']

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path+fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked = evoked.pick_channels(channel)
            power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            evoked_list_prep.append(power)

            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked = evoked.pick_channels(channel)
            power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            evoked_list_pca.append(power)

            input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked = evoked.pick_channels(channel)
            power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            evoked_list_ica.append(power)

            input_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/6 projections/"
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked = evoked.pick_channels(channel)
            power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            evoked_list_ssp6.append(power)

        averaged_prep = mne.grand_average(evoked_list_prep, interpolate_bads=False, drop_bads=False)
        averaged_pca = mne.grand_average(evoked_list_pca, interpolate_bads=False, drop_bads=False)
        averaged_ica = mne.grand_average(evoked_list_ica, interpolate_bads=False, drop_bads=False)
        averaged_ssp6 = mne.grand_average(evoked_list_ssp6, interpolate_bads=False, drop_bads=False)

        tmin = -0.3
        tmax = 0.5
        vmin = -400
        vmax = -240
        # fig, ax = plt.subplots(1, 5, figsize=[18, 6], gridspec_kw={"width_ratios": [10, 10, 10, 10, 1]})
        fig, axis = plt.subplots(2, 2, figsize=[12, 12], constrained_layout=True)
        ax = axis.flatten()
        averaged_prep.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                          axes=ax[0], show=False, colorbar=False, dB=True,
                          tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
        averaged_pca.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                          axes=ax[1], show=False, colorbar=False, dB=True,
                          tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
        averaged_ica.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                          axes=ax[2], show=False, colorbar=False, dB=True,
                          tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
        averaged_ssp6.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                          axes=ax[3], show=False, colorbar=False, dB=True,
                          tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
        # Add colorbar
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                            wspace=0.2, hspace=0.1)
        cbar_ax = fig.add_axes([0.83, 0.1, 0.01, 0.8])
        cb = fig.colorbar(ax[0].images[-1], cax=cbar_ax, shrink=0.6)
        # cb = im[-1].colorbar
        cb.set_label('Amplitude (dB)')
        # Add axis for white space
        # fig.subplots_adjust(bottom=0.25)
        # cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.01])
        # cb = fig.colorbar(ax[0].images[-1], cax=cbar_ax, shrink=0.6, orientation='horizontal')
        # cb.set_label('Amplitude (dB)')
        ax[0].set_title('Uncleaned')
        ax[0].set_xticklabels([])
        ax[0].set_xlabel(None)
        ax[1].set_title('PCA Cleaned')
        ax[1].set_yticklabels([])
        ax[1].set_ylabel(None)
        ax[1].set_xticklabels([])
        ax[1].set_xlabel(None)
        ax[2].set_title('ICA Cleaned')
        ax[3].set_title('SSP Cleaned')
        ax[3].set_yticklabels([])
        ax[3].set_ylabel(None)
        if cond_name == 'median':
            plt.suptitle(f"Time-Frequency Representation of the Cardiac Artefact\n"
                         f"Cervical Spinal Cord")
        else:
            plt.suptitle(f"Time-Frequency Representation of the Cardiac Artefact\n"
                         f"Lumbar Spinal Cord")
        fname = f"Heart_{cond_name}_SAB"
        plt.savefig(image_path+fname)
        plt.savefig(image_path+fname+'.pdf', bbox_inches='tight', format="pdf")
    plt.show()
