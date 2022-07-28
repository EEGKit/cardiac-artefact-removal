# Want to test reliability of each method to extract negative potentials
# Achieved by subsampling certain subjects and trial numbers
import numpy as np
import h5py
import pandas as pd
from scipy.stats import ttest_1samp as ttest
import matplotlib.pyplot as plt
import os


def reliability(data, noExperiment, trialSizes, subjectSizes):
    # Set up output array for p-values and effect sizes, 1000 x subject size x trial size
    p_vals = np.zeros((noExperiment, len(subjectSizes), len(trialSizes)))
    effect_sizes = np.zeros((noExperiment, len(subjectSizes), len(trialSizes)))

    df = pd.DataFrame(data)  # Convert 2d array to dataframe for ease of sampling
    for bIdx in np.arange(0, noExperiment):
        for s in np.arange(0, len(subjectSizes)):
            subjSample = df.sample(n=subjectSizes[s])  # Randomly sample number of subjects

            for t in np.arange(0, len(trialSizes)):
                y = subjSample.sample(n=trialSizes[t], axis='columns')
                # Want to ttest mean of each row - want a column vector
                _, p_vals[bIdx, s, t] = ttest(a=y.mean(axis=1).to_numpy(), popmean=0)
                y_arr = y.to_numpy()
                effect_sizes[bIdx, s, t] = np.mean(np.mean(y_arr, axis=1)/np.std(np.mean(y_arr, axis=1)))

    # Plot data and look at results
    probs = np.zeros((len(subjectSizes), len(trialSizes)))
    for t in np.arange(0, len(trialSizes)):
        for s in np.arange(0, len(subjectSizes)):
            probs[s, t] = sum(p_vals[:, s, t] < 0.05)/len(p_vals[:, s, t])

    plt.figure()
    plt.plot(trialSizes, probs.T)
    plt.xlim([0, 1000])
    plt.ylim([0, 1.2])
    plt.xlabel('Number of Trials')
    plt.ylabel('Proportion of Significant Experiments')
    plt.legend(['N=4', 'N=8', 'N=12', 'N=16', 'N=20'])


if __name__ == '__main__':
    figure_path = '/data/p_02569/Reliability Assessment/'
    os.makedirs(figure_path, exist_ok=True)
    which_method = {'Prep': True,
                    'PCA': True,
                    'Post-ICA': True,
                    'SSP_5': True,
                    'SSP_6': True}

    for i in np.arange(0, len(which_method)):
        method = list(which_method.keys())[i]
        if which_method[method]:

            if method == 'Prep':
                file_path = '/data/pt_02569/tmp_data/prepared_py/'
                file_name = 'amplitudes.h5'
            elif method == 'PCA':
                file_path = '/data/pt_02569/tmp_data/ecg_rm_py/'
                file_name = 'amplitudes.h5'
            elif method == 'Post-ICA':
                file_path = '/data/pt_02569/tmp_data/ica_py/'
                file_name = 'amplitudes.h5'
            elif method == 'SSP_5':
                file_path = '/data/p_02569/SSP/'
                file_name = 'amplitudes_5.h5'
            elif method == 'SSP_6':
                file_path = '/data/p_02569/SSP/'
                file_name = 'amplitudes_6.h5'

            keywords = ['amp_med', 'amp_tib']
            with h5py.File(file_path+file_name, "r") as infile:
                # Get the data
                amp_med = infile[keywords[0]][()]  # 36 x 2000
                amp_tib = infile[keywords[1]][()]  # 36 x 2000

            noexp = 1000  # Repeat 1000 times
            # trials = np.arange(5, np.shape(amp_med)[1], 10)  # Sample this many trials
            trials = np.arange(5, 1000, 10)  # Sample this many trials
            subjects = [4, 8, 12, 16, 20]  # Sample this many subjects
            for condition in ['Median Stimulation', 'Tibial Stimulation']:
                figure_name = f'Reliability_{condition}.png'
                reliability(amp_med, noexp, trials, subjects)
                plt.title(f'Reliability Assessment, {method}, {condition}')
                plt.savefig(figure_path+figure_name)
                plt.clf()

