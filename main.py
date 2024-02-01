###########################################################
# Emma Bailey, 08/11/2021
# This is a wrapper script for the pipeline
###########################################################

from rm_heart_artefact import *
from rm_heart_artefact_tukey import *
from SSP import *
from import_data import *
from epoch_data import *
from Archive.add_qrs_asevent import *
from ICA import run_ica
from ICA_anterior import run_ica_anterior
from ICA_separated import run_ica_separatepatches
from run_CCA import run_CCA

if __name__ == '__main__':
    ######## Want to import the data? ############
    import_d = False  # Prep work
    pchip_interpolation = False  # If true import with pchip, otherwise use linear interpolation

    ######## Want to use PCA_OBS to remove the heart artefact? #########
    heart_removal = False  # Heart artefact removal
    pchip = False  # Whether to use pchip prepared data or not
    heart_removal_tukey = False  # Fitted artefact multiplied by tukey window

    ######## Want to cut epochs from the PCA_OBS corrected data? ########
    cut_epochs = False  # Epoch the data according to relevant event

    ######### Want to clean the heart artefact using SSP? ########
    SSP_flag = False  # Heart artefact removal by SSP

    ######### Want to perform ICA to clean the heart artefact? ########
    ica = False  # Run ICA on the 'dirty'
    # choose_limited should be false - SNR is worse if it's true, over 95% residual intensity and inps under 1.4
    choose_limited = False  # If true only take the top 4 ICA components from find_bads_ecg
    ica_anterior = False  # Run ICA on anteriorly rereferenced data
    ica_separate_patches = False  # Run ICA on lumbar and cervical patches separately

    ######## Want to use Canonical Correlation Analysis to clean the heart artefact? ########
    CCA_flag = True  # Run CCA on data (from all methods)

    n_subjects = 36  # Number of subjects
    subjects = np.arange(1, 3)  # 1 through 36 to access subject data
    # subjects = [1]
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
                import_data(subject, condition, srmr_nr, sampling_rate, pchip_interpolation)

    ## To remove heart artifact via PCA_OBS ##
    if heart_removal:
        for subject in subjects:
            for condition in conditions:
                rm_heart_artefact(subject, condition, srmr_nr, sampling_rate, pchip)
                # If pchip is true, uses data where stim artefact was removed by pchip

    ## To remove heart artifact via PCA_OBS and tukey window ##
    if heart_removal_tukey:
        for subject in subjects:
            for condition in conditions:
                rm_heart_artefact_tukey(subject, condition, srmr_nr, sampling_rate, pchip)

    ## To cut epochs around triggers - only for PCA_OBS cleaned data here ##
    ## To cut epochs for all data types use get_epoched  ##
    if cut_epochs:
        for subject in subjects:
            for condition in conditions:
                epoch_data(subject, condition, srmr_nr, sampling_rate)

    ## Run ICA on data ##
    if ica:
        for subject in subjects:
            for condition in conditions:
                run_ica(subject, condition, srmr_nr, sampling_rate, choose_limited)

    if ica_anterior:
        for subject in subjects:
            for condition in conditions:
                run_ica_anterior(subject, condition, srmr_nr, sampling_rate)

    if ica_separate_patches:
        for subject in subjects:
            for condition in conditions:
                run_ica_separatepatches(subject, condition, srmr_nr, sampling_rate)

    ## To remove heart artifact using SSP method in MNE ##
    if SSP_flag:
        for subject in subjects:
            for condition in conditions:
                apply_SSP(subject, condition, srmr_nr, sampling_rate)

    ## Run CCA on the data ##
    data_strings = ['Prep', 'PCA']  # ' Post-ICA' no longer used, 'ICA' not used due to how decimated the signal is
    n = 5
    if CCA_flag:
        for data_string in data_strings:
            for subject in subjects:
                for condition in conditions:
                    run_CCA(subject, condition, srmr_nr, data_string, n)

        # Treat SSP separately
        data_string = 'SSP'
        for n in np.arange(5, 7):  # 21
            for subject in subjects:
                for condition in conditions:
                    run_CCA(subject, condition, srmr_nr, data_string, n)
