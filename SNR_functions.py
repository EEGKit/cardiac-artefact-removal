import mne
import numpy as np
from statistics import mean


def evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs):
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                        baseline=tuple(iv_baseline))

    if reduced_epochs and trigger_name == 'Median - Stimulation':
        epochs = epochs[900:1100]
    elif reduced_epochs and trigger_name == 'Tibial - Stimulation':
        epochs = epochs[800:1200]

    evoked = epochs.average()

    return evoked


# Takes an evoked response and calculates the SNR
def calculate_SNR_evoked(evoked, cond_name, iv_baseline, reduced_window):
    # Drop TH6 and ECG from channels from average rereferenecd channels
    # For anterior rereference remove those channels instead
    if 'TH6' in evoked.ch_names:
        evoked.drop_channels(['TH6', 'ECG'])
    elif 'AL' in evoked.ch_names:
        evoked.drop_channels(['AL', 'ECG'])
    elif 'AC' in evoked.ch_names:
        evoked.drop_channels(['AC', 'ECG'])

    # Want to only check channels relevant to potential being triggered
    # Tibial centred around 22ms - take 10ms on either end
    # Median centred around 13ms - take 5ms on either end
    # Trying smaller window as per Falk's suggestion - 2ms on either end
    if cond_name == 'tibial':
        channels = ['S23', 'L1', 'S31']
        if reduced_window:
            start = 20/1000
            end = 24/1000
        else:
            start = 12 / 1000
            end = 32 / 1000

    elif cond_name == 'median':
        channels = ['S6', 'SC6', 'S14']
        if reduced_window:
            start = 11/1000
            end = 15/1000
        else:
            start = 8 / 1000
            end = 18 / 1000

    # Want to select the channel with the maximal signal evoked in the correct time window out of the relevant channels
    peak_amplitude = 0
    peak_channel = 'L4'
    peak_latency = 0

    for ch in channels:
        # evoked.plot(picks=ch_name)
        evoked_channel = evoked.copy().pick_channels([ch])

        # Check data in channel actually has neg values - sub 24 in the baseline ICA, one channel doesn't
        # That's why this check was added
        time_idx = evoked_channel.time_as_index([start, end])
        data = evoked_channel.data[0, time_idx[0]:time_idx[1]]  # only one channel, time period

        # Extract negative peaks
        if np.any(data < 0):
            _, latency, amplitude = evoked_channel.get_peak(ch_type=None, tmin=start, tmax=end, mode='neg',
                                                            time_as_index=False, merge_grads=False,
                                                            return_amplitude=True)
        # If there are no negative values, insert dummys
        else:
            latency = np.nan
            amplitude = np.nan
            ch = 'L4'

        if abs(amplitude) > peak_amplitude:
            peak_amplitude = abs(amplitude)
            peak_channel = ch
            peak_latency = latency

    # If at the end the peak amplitude hasn't updated from 0
    if peak_amplitude == 0:
        peak_amplitude = np.nan
        peak_channel = 'L4'
        peak_latency = np.nan

    # Now get the std in the baseline period of the selected channel
    print(peak_channel)
    peak_evoked = evoked.copy().pick_channels([peak_channel])
    iv_baseline_idx = peak_evoked.time_as_index([iv_baseline[0], iv_baseline[1]])
    # print(evoked.data[:2, :3])  # first 2 channels, first 3 timepoints - from tutorial
    base = peak_evoked.data[0, iv_baseline_idx[0]:iv_baseline_idx[1]]  # only one channel, baseline time period
    # Sanity check to make sure this is getting the right data and timepoints
    # plt.plot(evoked_channel.data[0, :])
    # plt.show()
    stan_dev = np.std(base, axis=0)

    # Compute and save snr
    snr = abs(peak_amplitude) / abs(stan_dev)

    return snr, peak_channel

# Takes an evoked response and calculates the SNR
def calculate_SNR_evoked_cca(evoked, cond_name, iv_baseline, reduced_window, selected_components):
    # Drop TH6 and ECG from channels from average rereferenecd channels
    # For anterior rereference remove those channels instead
    if 'TH6' in evoked.ch_names:
        evoked.drop_channels(['TH6', 'ECG'])
    elif 'AL' in evoked.ch_names:
        evoked.drop_channels(['AL', 'ECG'])
    elif 'AC' in evoked.ch_names:
        evoked.drop_channels(['AC', 'ECG'])

    # Want to only check channels relevant to potential being triggered
    # Potentials are 1-2ms later in this second dataset
    # Taking reduced tibial window per falks suggestion
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

