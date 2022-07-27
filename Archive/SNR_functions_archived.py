# Holds archived functions previously used in SNR calculations

import mne
import numpy as np


# Don't use SNR for CCA data anymore
def calculate_SNR_evoked_cca(evoked, cond_name, iv_baseline, reduced_window, selected_components):
    # Drop ECG and Reference channels
    if 'TH6' in evoked.ch_names:
        evoked.drop_channels(['TH6', 'ECG'])
    elif 'AL' in evoked.ch_names:
        evoked.drop_channels(['AL', 'ECG'])
    elif 'AC' in evoked.ch_names:
        evoked.drop_channels(['AC', 'ECG'])

    # Select time window based on condition
    if cond_name == 'tibial':
        if reduced_window:
            start = 20/1000
            end = 24/1000
        else:
            start = 21 / 1000
            end = 29 / 1000

    elif cond_name == 'median':
        if reduced_window:
            start = 11/1000
            end = 15/1000
        else:
            start = 8 / 1000
            end = 18 / 1000

    # Extract data in relevant time period
    time_idx = evoked.time_as_index([start, end])
    data = np.mean(evoked.data[:selected_components, time_idx[0]:time_idx[1]], axis=0)  # first n channels

    # Get the minimum in that time frame and the latency
    amplitude = np.amin(data)
    index = np.where(data == np.amin(data))[0] + time_idx[0]
    latency = evoked.times[index]
    ch = 'Cor1'  # Essentially just  dummy variable to keep format the same

    # If amplitude is negative, leave it be, otherwise insert dummies
    if amplitude >= 0:
        latency = np.nan
        amplitude = np.nan
        ch = 'Cor9'

    # Now get the std in the baseline period of the selected channel
    iv_baseline_idx = evoked.time_as_index([iv_baseline[0], iv_baseline[1]])
    base = np.mean(evoked.data[:selected_components, iv_baseline_idx[0]:iv_baseline_idx[1]], axis=0)  # first n channels

    stan_dev = np.std(base, axis=0)

    # Compute and save snr
    snr = abs(amplitude) / abs(stan_dev)

    return snr, ch


# Don't use this - use metrics from EEG-fMRI research
def calculate_heart_SNR_evoked(evoked, cond_name, iv_baseline, reduced_window):
    snr = []

    if reduced_window:
        start = -0.005
        end = 0.005
    else:
        start = -0.02
        end = 0.02

    # Drop TH6 and ECG from channels from average rereferenecd channels
    # For anterior rereference remove those channels instead
    if 'TH6' in evoked.ch_names:
        evoked.drop_channels(['TH6', 'ECG'])
    elif 'AL' in evoked.ch_names:
        evoked.drop_channels(['AL', 'ECG'])
    elif 'AC' in evoked.ch_names:
        evoked.drop_channels(['AC', 'ECG'])

    for ch in evoked.ch_names:
        evoked_channel = evoked.copy().pick_channels([ch])

        time_idx = evoked_channel.time_as_index([start, end])
        data = evoked_channel.data[0, time_idx[0]:time_idx[1]]  # only one channel, time period

        # Extract positive peaks
        if np.any(data > 0):
            _, latency, amplitude = evoked_channel.get_peak(ch_type=None, tmin=start, tmax=end, mode='abs',
                                                            time_as_index=False, merge_grads=False,
                                                            return_amplitude=True)
        # If there are no positive values, insert dummy
        else:
            amplitude = np.nan

        # Now get the std in the baseline period of the channels
        iv_baseline_idx = evoked_channel.time_as_index([iv_baseline[0], iv_baseline[1]])
        # print(evoked.data[:2, :3])  # first 2 channels, first 3 timepoints - from tutorial
        base = evoked_channel.data[0, iv_baseline_idx[0]:iv_baseline_idx[1]]  # only one channel, baseline time period
        # Sanity check to make sure this is getting the right data and timepoints
        # plt.plot(evoked_channel.data[0, :])
        # plt.show()
        stan_dev = np.std(base, axis=0)

        # Compute and save snr
        current_snr = abs(amplitude) / abs(stan_dev)
        snr.append(current_snr)

    snr_average = np.nanmean(snr)

    return snr_average


# Don't use this - use metrics from EEG-fMRI research
def calculate_heart_SNR_evoked_ch(evoked, cond_name, iv_baseline, reduced_window):
    snr = []

    if reduced_window:
        start = -0.005
        end = 0.005
    else:
        start = -0.02
        end = 0.02

    if cond_name == 'tibial':
        channels = ['S23', 'L1', 'S31']
        mode = 'pos'
    elif cond_name == 'median':
        channels = ['S6', 'SC6', 'S14']
        mode = 'neg'

    # Drop TH6 and ECG from channels from average rereferenced channels
    # For anterior rereference remove those channels instead
    if 'TH6' in evoked.ch_names:
        evoked.drop_channels(['TH6', 'ECG'])
    elif 'AL' in evoked.ch_names:
        evoked.drop_channels(['AL', 'ECG'])
    elif 'AC' in evoked.ch_names:
        evoked.drop_channels(['AC', 'ECG'])

    for ch in channels:
        evoked_channel = evoked.copy().pick_channels([ch])

        time_idx = evoked_channel.time_as_index([start, end])
        data = evoked_channel.data[0, time_idx[0]:time_idx[1]]  # only one channel, time period

        # Extract positive peaks
        if np.any(data > 0):
            _, latency, amplitude = evoked_channel.get_peak(ch_type=None, tmin=start, tmax=end, mode=mode,
                                                            time_as_index=False, merge_grads=False,
                                                            return_amplitude=True)
        # If there are no positive values, insert dummy
        else:
            amplitude = np.nan

        # Now get the std in the baseline period of the channels
        iv_baseline_idx = evoked_channel.time_as_index([iv_baseline[0], iv_baseline[1]])
        # print(evoked.data[:2, :3])  # first 2 channels, first 3 timepoints - from tutorial
        base = evoked_channel.data[0, iv_baseline_idx[0]:iv_baseline_idx[1]]  # only one channel, baseline time period
        # Sanity check to make sure this is getting the right data and timepoints
        # plt.plot(evoked_channel.data[0, :])
        # plt.show()
        stan_dev = np.std(base, axis=0)

        # Compute and save snr
        current_snr = abs(amplitude) / abs(stan_dev)
        snr.append(current_snr)

    snr_average = np.nanmean(snr)

    return snr_average

