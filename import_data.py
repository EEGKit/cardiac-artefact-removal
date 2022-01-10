##########################################################################################
#                               This Script
# 1) imports the blocks based on the condition name in EEGLAB form from the BIDS directory
# 2) removes the stimulus artifact iv: -1.5 to 1.5 ms, for ESG use -7 to 7s
# 3) downsample the signal to 1000 Hz, r peak detection performed, save location of r-peaks separately
# 4) Append mne raws of the same condition
# 5) saves the new raw structure and the QRS events detected
# Emma Bailey, November 2021
##########################################################################################

# Import necessary packages
import mne
from get_conditioninfo import *
from get_channels import *
from scipy.io import loadmat
import h5py
import os
import glob


def import_data(subject, condition, srmr_nr, sampling_rate):
    # Set this to true if working with only esg data
    esg_flag = True

    # Set paths
    subject_id = f'sub-{str(subject).zfill(3)}'
    save_path = "../tmp_data/prepared_py/" + subject_id + "/esg/prepro/"  # Saving to prepared_py
    input_path = "/data/p_02068/SRMR1_experiment/bids/" + subject_id + "/eeg/"  # Taking data from the bids folder
    cfg_path = "/data/pt_02569/scripts/"  # Contains important info about experiment
    montage_name = 'standard-10-5-cap385_added_mastoids.elp'
    os.makedirs(save_path, exist_ok=True)

    # Get the condition information based on the condition read in
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    stimulation = condition - 1

    # Set interpolation window (different for eeg and esg data, both in seconds)
    tstart_esg = -0.007
    tmax_esg = 0.007

    tstart_eeg = -0.0015
    tmax_eeg = 0.006

    # File names - * allows you to match to a pattern rather than absolute
    # File names are related to the condition read in
    search = input_path + subject_id + '*' + cond_name + '*.set'
    cond_files = glob.glob(search)
    cond_files = sorted(cond_files)  # Arrange in order from lowest to highest value
    nblocks = len(cond_files)

    # Find out which channels are which, include ECG, exclude EOG
    eeg_chans, esg_chans, bipolar_chans = get_channels(subject_nr=subject, includesEcg=True, includesEog=False, study_nr=srmr_nr)

    ####################################################################
    # Extract the raw data for each block, remove stimulus artefact, downsample, concatenate, detect ecg, and then save
    ####################################################################
    # Looping through each condition and each subject in main.py
    # So in here, we're only dealing with one condition at a time, loop through however many blocks of said condition
    for iblock in np.arange(0, nblocks):
        # load data - need to read in files from EEGLAB format in bids folder
        fname = cond_files[iblock]
        raw = mne.io.read_raw_eeglab(fname, eog=(), preload=True, uint16_codec=None, verbose=None)

        # If you only want to look at esg channels, drop the rest
        if esg_flag == True:
            raw.pick_channels(esg_chans)

        # Interpolate required channels
        # Only interpolate tibial, medial and alternating (conditions 2, 3, 4 ; stimulation 1, 2, 3)
        if stimulation != 0:
            srate_basic = 1000 # Perform all analysis at 1000Hz

            if esg_flag == False:
                # Read in the array of electrodes from file
                montage = mne.channels.read_custom_montage(cfg_path + montage_name)

                # fits channel locations to data
                raw.set_montage(montage, on_missing="ignore")
                # Have to use ignore as the montage only includes EEG head channels (can't work with esg)
                # Quirk: For some reason it excludes FPz but we move

            # events contains timestamps with corresponding event_id
            # event_dict returns the event/trigger names with their corresponding event_id
            events, event_dict = mne.events_from_annotations(raw)

            # Fetch the event_id based on whether it was tibial/medial stimulation (trigger name)
            trigger_name = set(raw.annotations.description)

            # Acts in place to edit raw via linear interpolation to remove stimulus artefact
            # Need to loop as for alternating, there are 2 trigger names and event_ids at play
            for j in trigger_name:
                if esg_flag==True:
                    mne.preprocessing.fix_stim_artifact(raw, events=events, event_id=event_dict[j], tmin=tstart_esg,
                                                        tmax=tmax_esg, mode='linear', stim_channel=None)

                elif esg_flag==False:
                    mne.preprocessing.fix_stim_artifact(raw, events=events, event_id=event_dict[j], tmin=tstart_eeg,
                                                        tmax=tmax_eeg, mode='linear', stim_channel=None)

                else:
                    print('Flag has not been set - indicate if you are working with eeg or esg channels')

        # Downsample the data
        raw.resample(srate_basic)  # resamples to srate_basic


        # Append blocks of the same condition
        if iblock == 0:
            raw_concat = raw
        else:
            mne.concatenate_raws([raw_concat, raw])

    # Detects 1258 events for first subject, medial condition versus 1270 in MATLAB
    # They don't seem a million miles off though
    # Detect ECG events (contains a band pass filter) but start 10s before first event tstart = first_real_event/sr-10.0
    ecg_events, ch_ecg, average_pulse = mne.preprocessing.find_ecg_events(raw_concat, event_id=999, ch_name='ECG',
                                                                            tstart=0, l_freq=5,
                                                                            h_freq=35, qrs_threshold='auto',
                                                                            filter_length='5s')

    # Read .mat file with QRS events
    input_path_m = "/data/pt_02569/tmp_data/prepared/"+subject_id+"/esg/prepro/"
    fname_m = f"raw_{sampling_rate}_spinal_{cond_name}"
    matdata = loadmat(input_path_m + fname_m + '.mat')
    QRSevents_m = matdata['QRSevents'][0]

    # Add qrs events as annotations
    qrs_event = [x / sampling_rate for x in QRSevents_m]  # Divide by sampling rate to make times
    duration = np.repeat(0.0, len(QRSevents_m))
    description = ['qrs'] * len(QRSevents_m)
    raw_concat.annotations.append(qrs_event, duration, description, ch_names=[esg_chans] * len(QRSevents_m))
    fname_save = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'

    # Should have a single file name per condition now (just medial/tibial)
    # Save data without stim artefact and downsampled to 1000
    raw_concat.save(os.path.join(save_path, fname_save), fmt='double', overwrite=True)

    # Save detected QRS events
    dataset_keyword = 'QRS'
    fn = save_path + 'noStimart_sr%s_%s_withqrs.h5' % (srate_basic, cond_name)
    with h5py.File(fn, "w") as outfile:
        outfile.create_dataset(dataset_keyword, data=ecg_events[:, 0])
