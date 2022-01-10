##########################################################################
# Comparing the outputs from subject 1 in the tibial and medial condition
# One set of files from MATLAB processing, the other from python
##########################################################################

import mne
import matplotlib.pyplot as plt

# Set paths to data
input_path_M = "/data/pt_02569/tmp_data/test_data/sub-001/esg/prepro/"
input_path_P = "/data/pt_02569/tmp_data/prepared_py/sub-001/esg/prepro/"

# File name to test
fname_M = 'noStimart_sr5000_median_1.set' # In her prep, she doesn't actually drop the EEG channels, so this is still 114
fname_P = 'noStimart_sr1000_median_1.fif' # This file only includes ESG channels now

# Import data
raw_M = mne.io.read_raw_eeglab(input_path_M + fname_M, eog=(), preload=True, uint16_codec=None, verbose=None)
raw_P = mne.io.read_raw_fif(input_path_P + fname_P)

# Extract events to compare
events_M, event_dict_M = mne.events_from_annotations(raw_M)
events_P, event_dict_P = mne.events_from_annotations(raw_P)
# Plot some channels raw data
# fig1 = mne.viz.plot_raw(raw_M.pick_channels(['FPz', 'S23']), duration=5, events=events_M)
# plt.show()
# fig2 = mne.viz.plot_raw(raw_P.pick_channels(['FPz', 'S23']), duration=5, events=events_P)
# plt.show()

print(len(raw_M.ch_names))
print(len(raw_P.ch_names))

# Compare PSD
fig1 = raw_M.plot_psd(fmax=100)
fig2 = raw_P.plot_psd(fmax=100)
plt.show()

# Try to extract just the data to work on
# raw_M.data
