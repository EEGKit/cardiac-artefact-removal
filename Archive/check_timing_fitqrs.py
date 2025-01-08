# Want to check how close the fit end and fit start window timings are to the stimulation timings in PCA

# Check how many qrs timings coincide with same

import mne
import numpy as np

subjects = [2, 3, 20]  # Subject 20 is about twice as bad as subject 22
cond_name = 'tibial'
numbers = []

# Get from annotations in raw file name
for subject in subjects:
    subject_id = f'sub-{str(subject).zfill(3)}'

    input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id
    fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # Events has shape (no. events, 3) - first column is sample number, third column is event number
    # event_id gives code to word of the event number
    events, event_id = mne.events_from_annotations(raw)

    # Fit end
    code_end = event_id['fit_end']
    where_end = np.where(events == code_end)[0]  # Gives sample no where this event number is found
    sample_no_end = events[where_end][:, 0]  # Extracts just the sample numbers of these events

    # Fit start
    code_start = event_id['fit_start']
    where_start = np.where(events == code_start)[0]  # Gives sample no where this event number is found
    sample_no_start = events[where_start][:, 0]  # Extracts just the sample numbers of these events

    # QRS
    code_qrs = event_id['qrs']
    where_qrs = np.where(events == code_qrs)[0]  # Gives sample no where this event number is found
    sample_no_qrs = events[where_qrs][:, 0]  # Extracts just the sample numbers of these events

    # Stim
    code_stim = event_id['Tibial - Stimulation']
    where_stim = np.where(events == code_stim)[0]  # Gives sample no where this event number is found
    sample_no_stim = events[where_stim][:, 0]  # Extracts just the sample numbers of these events

    # print(sample_no_stim)
    # print(np.shape(sample_no_stim))
    # print(type(sample_no_stim))
    # print(sample_no_end)
    # print(np.shape(sample_no_end))
    # print(type(sample_no_end))
    t = 300
    count = 0
    for val in sample_no_stim:
        temp = sample_no_qrs - val  # Gets the distance in samples between occurence of stimulation and this heartbeat
        number = np.sum(np.abs(temp) < t)  # Checks how many heartbeats are close enough to our target
        # count += number  # Add how many occurrences are close enough
        if number > 0:
            count += 1  # Add how many occurrences are close enough

    numbers.append(count)

print(subjects)
print(numbers)


