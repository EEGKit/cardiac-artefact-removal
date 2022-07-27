# Want to test reliability of each method to extract negative potentials
# Achieved by subsampling certain subjects and trial numbers
import numpy as np
import h5py
import pandas as pd
from scipy.stats import ttest_1samp as ttest
import matplotlib.pyplot as plt

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
            probs[s, t] = sum(i for i in p_vals[:, s, t] if i > 0.05)/(len(p_vals[:, s, t]))

    plt.figure()
    plt.plot(trialSizes, probs.T)
    plt.ylim([0, 1.2])
    plt.xlabel('Number of Trials')
    plt.ylabel('Proportion of Significant Experiments')
    plt.legend(['5', '10', '15', '20', '25'])
    plt.show()


if __name__ == '__main__':
    method_names = ['PCA']
    methods_todo = [True]

    for i in np.arange(0, len(method_names)):
        method = method_names[i]
        if methods_todo[i]:  # If this method is true, go through with the rest

            if method == 'PCA':
                file_path = '/data/pt_02569/tmp_data/ecg_rm_py/'
                file_name = 'amplitudes.h5'

            keywords = ['amp_med', 'amp_tib']
            with h5py.File(file_path+file_name, "r") as infile:
                # Get the data
                amp_med = infile[keywords[0]][()]  # 36 x 2000
                amp_tib = infile[keywords[1]][()]  # 36 x 2000

            noexp = 1000  # Repeat 1000 times
            trials = np.arange(5, np.shape(amp_med)[1], 10)  # Sample this many trials
            subjects = [5, 10, 15, 20, 25]  # Sample this many subjects
            reliability(amp_med, noexp, trials, subjects)

######################### MATLAB CODE ###################
# for bIdx = 1:noExperiment % simulate a certain number of experiments
#     for s = 1:numel(subjectSizes)
#         clear subjSample
#         subjSample = datasample(myData, subjectSizes(s),1); %sample certain number of subjects
#         for t = 1:numel(trialSizes)
#             y = datasample(subjSample, trialSizes(t), 2); %sample certain number of trials from each subject
#             [~,p(bIdx,s,t)] = ttest(mean(y,2)); %get the p value from this sub-sample (sampled subjects & sampled trials)
#             effectSizes(bIdx,s,t) = mean(mean(y,2))/std(mean(y,2)); %calculate effect size
#             clear y
#         end
#     end
# end
# end
# % plot the data and look at results
# for t = 1:numel(trialSizes)
#     for s = 1:numel(subjectSizes)
#         Probs(s,t) = sum(p(:,s,t) <0.05)/numel(p(:,s,t));
#     end
# end
# figure; hold on; plot(trialSizes,Probs', 'LineWidth',2)
# ylim([0 1.2])
# ylabel('Proportion of significant experiments')
# xlabel('Number of trials')
# legend({'N = 5', 'N = 10', 'N = 15', 'N = 20', 'N = 24'})
#
