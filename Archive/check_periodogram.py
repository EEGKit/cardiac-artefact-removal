# Checking periodogram from baseline ICA due to weird INPS results

from scipy import signal
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

channel = 'SC6'
sf = 1000
cond_name = 'median'
subject = 6
subject_id = f'sub-{str(subject).zfill(3)}'
cfg_path = "/data/pt_02569/"  # Contains important info about experiment
cfg = loadmat(cfg_path + 'cfg.mat')
notch_freq = cfg['notch_freq'][0]
esg_bp_freq = cfg['esg_bp_freq'][0]

Prepared = False
ICA = False
PCA = True

if ICA:
    input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id
    fname = f"clean_baseline_ica_auto_{cond_name}.fif"
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

elif Prepared:
    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
    raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sf}_{cond_name}_withqrs.fif", preload=True)

elif PCA:
    input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id
    fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

# Extract data
data = raw.get_data(picks=[channel])

# Define window length (4 seconds)
win = 4 * sf
freqs, psd = signal.welch(data, sf, nperseg=win)
psd = psd.reshape(-1)

# Plot the power spectrum
sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 4))
plt.plot(freqs, psd, color='k', lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
# plt.xlim([0, freqs.max()])
plt.xlim([0, 100])
sns.despine()
plt.show()
