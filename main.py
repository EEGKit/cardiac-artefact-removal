###########################################################
# Creating Python Equivalent of Birgit Nierula's MATLAB Code
# Emma Bailey, 08/11/2021
# This is essentially a wrapper script for the pipeline
###########################################################

import numpy as np
import mne
from get_conditioninfo import *
from rm_heart_artefact import *
from SSP import *
from import_data import *
from epoch_data import *
from Post_ICA import *
from add_qrs_asevent import *
from ICA import run_ica
from run_CCA import run_CCA

if __name__ == '__main__':

    ## Pick which scripts you want to run ##
    import_d = False  # Prep work
    heart_removal = False  # Heart artefact removal
    cut_epochs = False  # Epoch the data according to relevant event
    SSP_flag = False  # Heart artefact removal by SSP
    post_ica = False  # Run ICA after already running PCA_OBS
    ica = False  # Run ICA on the 'dirty' data as a baseline comparison
    CCA_flag = True  # Run CCA on data (from all methods)

    n_subjects = 36  # Number of subjects
    # Testing with just subject 1 at the moment
    subjects = np.arange(1, 37)  # (1, 2) # 1 through 36 to access subject data
    srmr_nr = 1  # Experiment Number
    conditions = [2, 3]  # Conditions of interest
    sampling_rate = 1000

    ############################################
    # Import Data from BIDS directory
    # Select channels to analyse
    # Remove stimulus artefact if needed
    # Downsample and concatenate blocks of the same conditions
    # Detect QRS events
    # Save the new data and the QRS events
    ############################################
    if import_d:
        for subject in subjects:
            for condition in conditions:
                import_data(subject, condition, srmr_nr, sampling_rate)

    ## To remove heart artifact via PCA_OBS ##
    if heart_removal:
        for subject in subjects:
            for condition in conditions:
                rm_heart_artefact(subject, condition, srmr_nr, sampling_rate)


    ## To cut epochs around triggers ##
    if cut_epochs:
        for subject in subjects:
            for condition in conditions:
                epoch_data(subject, condition, srmr_nr, sampling_rate)

    ## Run post ICA script
    if post_ica:
        for subject in subjects:
            for condition in conditions:
                run_post_ica(subject, condition, srmr_nr, sampling_rate)


    ## Run ICA on the 'dirty' data
    if ica:
        for subject in subjects:
            for condition in conditions:
                run_ica(subject, condition, srmr_nr, sampling_rate)

    ## To remove heart artifact using SSP method in MNE ##
    if SSP_flag:
        for subject in subjects:
            for condition in conditions:
                apply_SSP(subject, condition, srmr_nr, sampling_rate)

    ## Run CCA on the data
    # The functionality of CCA is different in dataset 1 versus dataset 2
    # These scripts DO NOT work the same - be wary when applying which run_cca.py you wish to use
    data_strings = ['Prep', 'Post-ICA', 'PCA']  #, 'ICA' - ICA not working due to how decimated the signal is
    n = 5
    if CCA_flag:
        for data_string in data_strings:
            for subject in subjects:
                for condition in conditions:
                    run_CCA(subject, condition, srmr_nr, data_string, n)

        # Treat SSP separately
        data_string = 'SSP'
        for n in np.arange(5, 21):
            for subject in subjects:
                for condition in conditions:
                    run_CCA(subject, condition, srmr_nr, data_string, n)
