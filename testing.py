############################ Mess Script #########################################
# Originally for testing the functionality of pchip in python
# Became a quick sanity check script for anything required
# NOT relevant to the project
###################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py
from scipy.interpolate import pchip
# x = np.concatenate([np.arange(0, 6, 1), np.arange(12, 21, 1)]) # Values to keep unchanged
# y = np.random.rand(15) # Data point ys associated with x's above
# xi = np.linspace(5, 12, num=8).astype('int') # Where you want to interpolate between, with linspace it assumes there are 50 points within this range
# yi = pchip(x, y)(xi) # Returns just the interpolated y values
# print(xi)
# print(x)
# print(yi)
# print(y)
# plt.plot(x, y, 'r.', xi, yi)
# plt.show()

# Testing with more realistic values from my data
# intpol_window = [3308, 3320]
# n_samples_fit = 20
# fs = 1000
# x_interpol = np.arange(intpol_window[0], intpol_window[1] + 1, 1) # points to be interpolated in pt - the gap between the endpoints of the window
# x_fit = np.concatenate([np.arange(intpol_window[0] - n_samples_fit, intpol_window[0] + 1, 1),
#                         np.arange(intpol_window[1], intpol_window[1] + n_samples_fit + 1, 1)]) # Entire range of x values in this step (taking some number of samples before and after the window)
# y_fit = fitted_art[0, x_fit]
# y_interpol = pchip(x_fit, y_fit)(x_interpol)
# plt.plot(x_fit, y_fit, 'r.', x_interpol, y_interpol)
# plt.show()

# Read in the clean data and see how it looks
import mne
# Debugging channels ['S35', 'Iz', 'SC1', 'S3', 'SC6', 'S20', 'L1', 'L4']
# ch = ['SC1']
# fn = "/data/pt_02569/tmp_data/ecg_rm_py/sub-001/esg/prepro/data_clean_ecg_spinal_tibial.fif"
# raw = mne.io.read_raw_fif(fn, preload=True)
# data = raw.get_data(picks=ch)
# plt.plot(np.arange(0,len(data[0, :])), data[0, :])
# plt.title(f"Subject 1, channel {ch[0]}")
# plt.show()


# # Read in the cleaned data from matlab and python and plot overlayed
# folder_p = '/data/pt_02569/tmp_data/ecg_rm_py/sub-016/esg/prepro/'
# folder_m = '/data/pt_02569/tmp_data/ecg_rm/sub-016/esg/prepro/'
# matlab = 'cnt_clean_ecg_spinal_tibial.set'
# python = 'data_clean_ecg_spinal_tibial.fif'
#
# raw_m = mne.io.read_raw_eeglab(folder_m+matlab, eog=(), preload=True, uint16_codec=None, verbose=None)
# raw_p = mne.io.read_raw_fif(folder_p+python, preload=True)
# data_m = raw_m.get_data(picks=['S35'])
# data_p = raw_p.get_data(picks=['S35'])
# fig = plt.figure()
# plt.plot(np.arange(0,len(data_p[0, :])), data_p[0, :], 'm')
# plt.plot(np.arange(0,len(data_m[0, :])), data_m[0, :], 'k')
# plt.legend(['Python', 'MATLAB'])
# plt.title(f"Subject 16, channel S35")
# plt.show()

# # Plot epoch data as quick sanity check
# folder_p = '/data/pt_02569/tmp_data/epoched_py/sub-001/esg/prepro/'
# folder_m = '/data/pt_02569/tmp_data/epoched/sub-001/esg/prepro/'
# matlab = 'epo_cleanclean_median.set'
# python = 'epo_clean_median.fif'
#
# epochs_m = mne.read_epochs_eeglab(folder_m+matlab)
# epochs_p = mne.read_epochs(folder_p+python)
#
# channelNames = ['S35', 'Iz', 'SC1', 'S3', 'SC6', 'S20', 'L1', 'L4']
# # Can use combine='mean' to get mean across channels selected
# epochs_m['Median - Stimulation'].plot_image(picks=channelNames, vmin=-5, vmax=5)
# epochs_p['Median - Stimulation'].plot_image(picks=channelNames, vmin=-5, vmax=5)

# Read in file with qrs events added, epoch around the heart peaks and plot to see how it does
folder = '/data/pt_02569/tmp_data/prepared_py/sub-001/esg/prepro/'
file = 'noStimart_sr1000_median_withqrs.fif'

raw = mne.io.read_raw_fif(folder+file)

events, event_ids = mne.events_from_annotations(raw)
event_id_dict = {key: value for key, value in event_ids.items() if key == 'qrs'}
epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=-200/1000, tmax=200/1000)

evoked = epochs.average()

raw.plot()
evoked.plot()
plt.show()


