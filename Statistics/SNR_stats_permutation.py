# File to compute one way ANOVA and post-hoc testing on effect of method on SNR
import mne.stats
import pandas as pd
import numpy as np
import h5py
from itertools import combinations
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from bioinfokit.analys import stat
import pingouin
from statannotations.Annotator import Annotator
import os
from statsmodels.formula.api import ols
import statsmodels.api as sm

if __name__ == '__main__':
    test_assumptions = False
    run_anova = False
    run_posthoc = True
    plot_graph = True

    # Read in results files and format in a pandas dataframe
    # Set file locations
    fname = 'snr.h5'
    prep_path = '/data/pt_02569/tmp_data/prepared_py/'
    pca_path = '/data/pt_02569/tmp_data/ecg_rm_py/'
    ica_path = '/data/pt_02569/tmp_data/baseline_ica_py/'
    postica_path = '/data/pt_02569/tmp_data/ica_py/'
    ssp_path = '/data/p_02569/SSP/'
    figure_path = '/data/p_02569/StatsGraphs_Dataset1/'
    os.makedirs(figure_path, exist_ok=True)

    ################################# Make Dataframe ###################################
    file_paths = [prep_path, pca_path, ica_path, postica_path, ssp_path]
    names = ['Prep', 'PCA', 'ICA', 'Post-ICA', 'SSP']
    names_indf = ['Prep', 'PCA', 'ICA', 'Post-ICA', 'SSP_5', 'SSP_6']
    # Pull each subjects value out
    keywords = ['snr_med', 'snr_tib']
    count = 0
    for file_path in file_paths:
        with h5py.File(file_path+fname, "r") as infile:
            if file_path == prep_path:
                snr_med = infile[keywords[0]][()].reshape(-1)
                data_med = {'Prep': snr_med}
                df_med = pd.DataFrame(data_med, index=np.arange(1, 37))

                snr_tib = infile[keywords[1]][()].reshape(-1)
                data_tib = {'Prep': snr_tib}
                df_tib = pd.DataFrame(data_tib, index=np.arange(1, 37))

            elif file_path == ssp_path:
                # These have shape (n_subjects, n_projectors)
                snr_med = infile[keywords[0]][()]
                df_med[f'{names[count]}_5'] = snr_med[:, 0]
                df_med[f'{names[count]}_6'] = snr_med[:, 1]

                snr_tib = infile[keywords[1]][()]
                df_tib[f'{names[count]}_5'] = snr_tib[:, 0]
                df_tib[f'{names[count]}_6'] = snr_tib[:, 1]

            else:
                # Get the data
                snr_med = infile[keywords[0]][()].reshape(-1)
                df_med[names[count]] = snr_med

                snr_tib = infile[keywords[1]][()].reshape(-1)
                df_tib[names[count]] = snr_tib

            count += 1

    #################################### Dataframe of Differences ###########################
    # Test
    # df = df_med.dropna()  # Drop NA- discuss with FALK
    # df = df_med.fillna(0)  # Fill NA values with 0
    df = df_med.fillna(df_med.mean())  # Replace with mean of column

    cc = list(combinations(df.columns, 2))
    df = pd.concat([df[c[1]].sub(df[c[0]]) for c in cc], axis=1, keys=cc)
    df.columns = df.columns.map('-'.join)
    arr = df.to_numpy()

    print(df_med)
    print(df)
    print(np.shape(arr))

    T_obs, p_values, H0 = mne.stats.permutation_t_test(arr, n_permutations=1000, n_jobs=36)
    print(p_values)

    formatted_pvals = {}
    colnames = df.columns
    for index in np.arange(0, len(p_values)):
        formatted_pvals.update({colnames[index]: p_values[index]})

    df_pvals = pd.DataFrame.from_dict(formatted_pvals, orient='index')
    print(df_pvals)
