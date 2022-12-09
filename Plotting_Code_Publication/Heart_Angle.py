# Want to get the distance from each stimulation point, to the R-peak before
# Get this in degrees
# Want an average 'angle' per participant and condition
# Then want to plot the average participant angle

import os
import numpy as np
import mne
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    cond_names = ['median', 'tibial']
    save_path = '/data/p_02569/Heart_Angle_D1/'
    os.makedirs(save_path, exist_ok=True)

    subjects = np.arange(1, 37)
    sampling_rate = 1000

    for cond_name in cond_names:
        subject_averages = []
        if cond_name == 'median':
            trigger_name = 'Median - Stimulation'
        elif cond_name == 'tibial':
            trigger_name = 'Tibial - Stimulation'

        for subj in subjects:
            subject_id = f'sub-{str(subj).zfill(3)}'

            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
            raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr1000_{cond_name}_withqrs.fif", preload=True)

            # Extract the stimulus event times
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            stim_events = [event.tolist() for event in events if event[2] == event_id_dict[trigger_name]]
            stim_times = np.asarray([stim_event[0]/sampling_rate for stim_event in stim_events])

            # Get the R-peak timings
            input_path_m = "/data/pt_02569/tmp_data/prepared/" + subject_id + "/esg/prepro/"
            fname_m = f"raw_{sampling_rate}_spinal_{cond_name}"
            matdata = loadmat(input_path_m + fname_m + '.mat')
            QRS_times = matdata['QRSevents'][0]/sampling_rate

            time_differences = []
            degree_differences = []
            for stim_time in stim_times:
                # Find time of R-peak before stimulus onset
                peak_time = QRS_times[QRS_times < stim_time].max()
                # As an index
                pos = QRS_times[QRS_times < stim_time].argmax()

                # Calcuate difference between stimulus onset and previous R-peak
                diff2peak = stim_time - peak_time
                time_differences.append(diff2peak)

                # Calculate relative position of the stimulus onset on the cardiac cycle
                stim_degree = 360*diff2peak/(QRS_times[pos+1]-QRS_times[pos])
                degree_differences.append(stim_degree)

            # Now get the average degrees for this subject
            if subj in [1, 6, 10, 19, 24, 32]:
                r = 2
                average_radians = np.deg2rad(np.asarray(degree_differences))
                ax = plt.subplot(111, projection='polar')
                for rad in average_radians:
                    plt.polar(rad, r, 'g.', linewidth=4)
                ax.set_theta_zero_location('N')
                ax.set_theta_direction(-1)  # clockwise
                ax.grid(False)
                ax.set_rticks([])
                ax.spines['polar'].set_visible(False)
                ax.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100) * r, color='black', linewidth=0.5)
                plt.ylim([0, r + 0.25])
                plt.title(f"{subject_id}, {cond_name}")
                plt.show()

            subject_averages.append(np.mean(degree_differences))

        # Now we have each subjects average, put on a polar plot
        print(subject_averages)
        r = 2
        average_radians = np.deg2rad(np.asarray(subject_averages))
        ax = plt.subplot(111, projection='polar')
        for rad in average_radians:
            plt.polar(rad, r, 'g.', linewidth=4)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)  # clockwise
        ax.grid(False)
        ax.set_rticks([])
        ax.spines['polar'].set_visible(False)
        ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*r, color='black', linewidth=0.5)
        plt.ylim([0, r+0.25])
        plt.title(f"All Subjects Average, {cond_name}")
        plt.show()
