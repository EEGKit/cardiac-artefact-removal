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
# folder = '/data/pt_02569/tmp_data/prepared_py/sub-001/esg/prepro/'
# file = 'noStimart_sr1000_median_withqrs.fif'
#
# raw = mne.io.read_raw_fif(folder+file)
#
# events, event_ids = mne.events_from_annotations(raw)
# event_id_dict = {key: value for key, value in event_ids.items() if key == 'qrs'}
# epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=-200/1000, tmax=200/1000)
#
# evoked = epochs.average()
#
# raw.plot()
# evoked.plot()
# plt.show()
#

# fn = f"/data/pt_02569/tmp_data/fit_end_timings.h5"
# with h5py.File(fn, 'r') as f:
#    data = f['median_timings_1']
#
# print(data)
# print(np.shape(data))

# Script to actually run CCA on the data

import os
import mne
import h5py
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from get_conditioninfo import get_conditioninfo


# Set variables
esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
             'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
             'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
             'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
             'S23', 'TH6']

# Tibial doesn't work with subject 1 - must have accidentally cropped it at some point to include only 2 channels
cond_name = 'median'
# cond_name = 'tibial'
trigger_name = 'Median - Stimulation'
# trigger_name = 'Tibial - Stimulation'

input_path = "/data/pt_02569/tmp_data/epoched/sub-001/esg/prepro/"
fname = f'epo_cleanclean_{cond_name}.set'

epochs = mne.read_epochs_eeglab(input_path + fname)

if cond_name == 'median':
    window_times = [8 / 1000, 18 / 1000]
    window = epochs.time_as_index(window_times)
elif cond_name == 'tibial':
    window_times = [18 / 1000, 28 / 1000]
    window = epochs.time_as_index(window_times)
else:
    print('Invalid condition name attempted for use')
    exit()

# Crop the epochs
epo_cca = epochs.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)

# Prepare matrices for cca
##### Average matrix
epo_av = epo_cca.copy().average().data.T
# Now want channels x observations matrix #np.shape()[0] gets number of trials
# Epo av is no_times x no_channels (10x40)
# Want to repeat this to form an array thats no. observations x no.channels (20000x40)
# Need to repeat the array, no_trials/times amount along the y axis
avg_matrix = np.tile(epo_av, (int((np.shape(epochs.get_data())[0])), 1))
avg_matrix = avg_matrix.T  # Need to transpose for correct form for function - channels x observations

##### Single trial matrix
epo_cca_data = epo_cca.get_data(picks=esg_chans)
epo_data = epochs.get_data(picks=esg_chans)

# 0 to access number of epochs, 1 to access number of channels
# channels x observations
no_times = int(window[1] - window[0])
# -1 means it automatically reshapes to what it needs after the other dimension is filled (with number of channels)
# Need to transpose to get it in the form CCA wants
st_matrix = np.swapaxes(epo_cca_data, 1, 2).reshape(-1, epo_cca_data.shape[1]).T
st_matrix_long = np.swapaxes(epo_data, 1, 2).reshape(-1, epo_data.shape[1]).T

# Run CCA
W_avg, W_st, r = spatfilt.CCA_data(avg_matrix, st_matrix)  # Getting the same shapes as matlab so far

if cond_name == 'median':
    W_st = W_st*-1  # Need to invert the weighting matrices to get the correct pattern, but not for tibial
all_components = len(r)

# Apply obtained weights to the long dataset (dimensions 40x9) - matrix multiplication
CCA_concat = st_matrix_long.T @ W_st[:, 0:all_components]
CCA_concat = CCA_concat.T

# Spatial Patterns
A_st = np.cov(st_matrix) @ W_st

# Reshape - (900, 2000, 9)
no_times_long = np.shape(epochs.get_data())[2]
no_epochs = np.shape(epochs.get_data())[0]

# Get into the same form as matlab and perform reshape as it does to be safe with reshape
CCA_comps = np.reshape(CCA_concat, (all_components, no_times_long, no_epochs), order='F')

# Now we have CCA comps, get the data in the axes format MNE likes (n_epochs, n_channels, n_times)
CCA_comps = np.swapaxes(CCA_comps, 0, 2)
CCA_comps = np.swapaxes(CCA_comps, 1, 2)
selected_components = 2

# Need to create an epoch data class to store the information
data = CCA_comps[:, 0:selected_components, :]
events = epochs.events
event_id = epochs.event_id
tmin = -0.2
sfreq = 1000

# Initialize an info structure
info = mne.create_info(
    ch_names=['Cor1', 'Cor2'],
    ch_types=['eeg', 'eeg'],
    sfreq=sfreq
)

cca_epochs = mne.EpochsArray(data, info, events, tmin, event_id)
cca_epochs['Median - Stimulation'].average().plot()
print(len(cca_epochs['qrs']))
exit()

plt.figure()
plt.plot(np.mean(CCA_comps[:, 0, :], axis=0), linewidth=0.6)
plt.show()

# # Declare class to hold pca information
# class CCAInfo():
#     def __init__(self):
#         pass
#
# # Instantiate class
# cca_info = CCAInfo()
#
# cca_info.average_mat = avg_matrix
# cca_info.single_mat = st_matrix*10**6
# cca_info.single_mat_long = st_matrix_long
# cca_info.epo_cca_data = epo_cca_data
# cca_info.CCA_concat = CCA_concat
# # cca_info.A_st = A_st
# cca_info.W_st = W_st
# cca_info.average_mat = avg_matrix
# cca_info.CCA_comps = CCA_comps
#
# dataset_keywords = [a for a in dir(cca_info) if not a.startswith('__')]
# fn = f"median_s-001_cca_info.h5"
# with h5py.File(fn, "w") as outfile:
#     for keyword in dataset_keywords:
#         outfile.create_dataset(keyword, data=getattr(cca_info, keyword))