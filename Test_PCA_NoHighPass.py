###########################################################
# Emma Bailey, 19/07/2023
# Test how no 0.9Hz high pass filter effects hump in lumbar data
###########################################################


from rm_heart_artefact_NoHighPass import *

if __name__ == '__main__':
    n_subjects = 36  # Number of subjects
    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    # subjects = [1]
    srmr_nr = 1  # Experiment Number
    conditions = [2, 3]  # Conditions of interest
    sampling_rate = 1000
    pchip = True

    for subject in subjects:
        for condition in conditions:
            rm_heart_artefact_NoHighPass(subject, condition, srmr_nr, sampling_rate, pchip)
