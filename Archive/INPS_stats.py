# File to compute one way ANOVA and post-hoc testing on effect of method on SNR

import pandas as pd
import numpy as np
import h5py
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
    run_anova = True
    run_posthoc = True
    plot_graph = True

    # Read in results files and format in a pandas dataframe
    # Set file locations
    fname = 'inps_yasa.h5'
    prep_path = '/data/pt_02569/tmp_data/prepared_py/'
    pca_path = '/data/pt_02569/tmp_data/ecg_rm_py/'
    ica_path = '/data/pt_02569/tmp_data/baseline_ica_py/'
    ssp_path = '/data/pt_02569/tmp_data/ssp_py/'
    figure_path = '/data/p_02569/Images/StatsGraphs_Dataset1/'
    os.makedirs(figure_path, exist_ok=True)

    # Necessary information to get relavant channel data
    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    median_pos = []
    tibial_pos = []
    for channel in ['S23', 'L1', 'S31']:
        tibial_pos.append(esg_chans.index(channel))
    for channel in ['S6', 'SC6', 'S14']:
        median_pos.append(esg_chans.index(channel))

    ################################# Make Dataframe ###################################
    file_paths = [prep_path, pca_path, ica_path, ssp_path]
    names = ['Prep', 'PCA', 'ICA', 'SSP_5', 'SSP_6']  # To make dataframe
    names_indf = ['PCA', 'ICA', 'SSP_5', 'SSP_6']  # To access columns in df
    # Pull each subjects value out
    keywords = ['pow_med', 'pow_tib']
    count = 0
    for file_path in file_paths:
        if file_path == ssp_path:
            fname = 'inps_yasa_5.h5'

        # Need the prep values to do all the divisions thereafter
        with h5py.File(file_path+fname, "r") as infile:
            if file_path == prep_path:  # Just extract values from file to use later
                val_prep_med = infile[keywords[0]][()]
                val_prep_tib = infile[keywords[1]][()]

            elif file_path == pca_path:  # Start the dataframe
                # Get the data
                val_med = infile[keywords[0]][()]
                res_med_current = (np.mean(val_prep_med[:, median_pos] / val_med[:, median_pos], axis=1))
                data_med = {'PCA': res_med_current}
                df_med = pd.DataFrame(data_med, index=np.arange(1, 37))

                val_tib = infile[keywords[1]][()]
                res_tib_current = (np.mean(val_prep_tib[:, tibial_pos] / val_tib[:, tibial_pos], axis=1))
                data_tib = {'PCA': res_tib_current}
                df_tib = pd.DataFrame(data_tib, index=np.arange(1, 37))

            else:
                # Get the data
                val_med = infile[keywords[0]][()]
                res_med_current = (np.mean(val_prep_med[:, median_pos] / val_med[:, median_pos], axis=1))
                df_med[names[count]] = res_med_current

                val_tib = infile[keywords[1]][()]
                res_tib_current = (np.mean(val_prep_tib[:, tibial_pos] / val_tib[:, tibial_pos], axis=1))
                df_tib[names[count]] = res_tib_current

            count += 1
    # Then just add SSP 6 to the last column
    fname = 'inps_yasa_6.h5'
    with h5py.File(ssp_path + fname, "r") as infile:
        val_med = infile[keywords[0]][()]
        res_med_current = (np.mean(val_prep_med[:, median_pos] / val_med[:, median_pos], axis=1))
        df_med['SSP_6'] = res_med_current

        val_tib = infile[keywords[1]][()]
        res_tib_current = (np.mean(val_prep_tib[:, tibial_pos] / val_tib[:, tibial_pos], axis=1))
        df_tib['SSP_6'] = res_tib_current

    # print('Median Averages')
    # print(df_med[['PCA', 'ICA', 'SSP_5', 'SSP_6']].mean())
    # print('Median Standard Deviation')
    # print(df_med[['PCA', 'ICA', 'SSP_5', 'SSP_6']].std())
    # print('Tibial Averages')
    # print(df_tib[['PCA', 'ICA', 'SSP_5', 'SSP_6']].mean())
    # print('Tibial Standard Deviation')
    # print(df_tib[['PCA', 'ICA', 'SSP_5', 'SSP_6']].std())
    # exit()

    ######################### Make Long Form Dataframe ################################
    df_med_melt = pd.melt(df_med.reset_index(), id_vars=['index'], value_vars=names_indf)
    df_med_melt.columns = ['subject', 'method', 'value']
    df_tib_melt = pd.melt(df_tib.reset_index(), id_vars=['index'], value_vars=names_indf)
    df_tib_melt.columns = ['subject', 'method', 'value']

    ################################# Test Assumptions ################################
    if test_assumptions:
        # Test Normality of Residuals
        df_med_melt_nonan = df_med_melt.dropna()
        df_tib_melt_nonan = df_tib_melt.dropna()
        stats_shap = pingouin.normality(df_med_melt_nonan, dv='value', group='method')
        print('Median Shapiro-Wilks')
        print(stats_shap)

        stats_shap = pingouin.normality(df_tib_melt_nonan, dv='value', group='method')
        print('Tibial Shapiro Wilks')
        print(stats_shap)

        # Test homogeneity - Non-normal so use Levenes test
        stats_lev = pingouin.homoscedasticity(df_med_melt_nonan, dv='value', group='method')
        print('Median Levenes')
        print(stats_lev)

        # Drop nan rows in df melt
        stats_lev = pingouin.homoscedasticity(df_tib_melt_nonan, dv='value', group='method')
        print('Tibial Levenes')
        print(stats_lev)

    ########################### Perform Kruskal-Wallis H-test ####################################
    if run_anova:
        # Non-parametric equivalent of ANOVA
        stats_m = pingouin.kruskal(data=df_med_melt, dv='value', between='method')
        stats_t = pingouin.kruskal(data=df_tib_melt, dv='value', between='method')
        print('Median Stimulation')
        print(stats_m)
        print('Tibial Stimulation')
        print(stats_t)

    ################################## Post-Hoc Mann-Whitney U Tests ####################################
    target = 'PCA'
    others = names_indf.copy()
    others.remove(target)

    if run_posthoc:
        # Non-parametric equivalent of paired sample t-tests
        # Nan-policy as pairwise only deletes relevant value, not whole list
        mw_m = pingouin.pairwise_tests(data=df_med_melt, dv='value', between='method', parametric=False,
                                       padjust='bonf', nan_policy='pairwise', return_desc=True)
        mw_t = pingouin.pairwise_tests(data=df_tib_melt, dv='value', between='method', parametric=False,
                                       padjust='bonf', nan_policy='pairwise', return_desc=True)
        print('Median Mann-Whitney')
        print(mw_m)
        print('Tibial Mann-Whitney')
        print(mw_t)

        ######################################## Extract P-Values ##################################
        # Want to set prep as target and then find it's relation to other methods to extract correct p-value
        p_values_med = []
        for name in others:
            for ind in np.arange(0, len(mw_m['A'])):
                if mw_m['A'].iloc[ind] == target and mw_m['B'].iloc[ind] == name:
                    p_values_med.append(mw_m["p-corr"].iloc[ind])
                elif mw_m['B'].iloc[ind] == target and mw_m['A'].iloc[ind] == name:
                    p_values_med.append(mw_m["p-corr"].iloc[ind])

        p_values_tib = []
        for name in others:
            for ind in np.arange(0, len(mw_m['A'])):
                if mw_t['A'].iloc[ind] == target and mw_t['B'].iloc[ind] == name:
                    p_values_tib.append(mw_t["p-corr"].iloc[ind])
                elif mw_t['B'].iloc[ind] == target and mw_t['A'].iloc[ind] == name:
                    p_values_tib.append(mw_t["p-corr"].iloc[ind])

        print('Corrected P-Values Median')
        print(p_values_med)
        print('Corrected P-Values Tibial')
        print(p_values_tib)

    ##################################### Plot with Stats ####################################
    if plot_graph:
        # Set up for annotations
        pairs = []
        for name in others:
            pairs.append((target, name))

        plotting_parameters_med = {
            'data': df_med_melt,
            'x': 'method',
            'y': 'value',
            'color': 'darkblue'
        }
        plotting_parameters_tib = {
            'data': df_med_melt,
            'x': 'method',
            'y': 'value',
            'color': 'deepskyblue'
        }
        conditions = ['Median Stimulation', 'Tibial Stimulation']
        plotting_parameters = [plotting_parameters_med, plotting_parameters_tib]
        p_values = [p_values_med, p_values_tib]

        for condition in conditions:
            if condition == 'Median Stimulation':
                ind = 0
            else:
                ind = 1
            # Create new plot
            ax = plt.figure()
            plt.yscale('log')

            # Plot with seaborn
            ax = sns.boxplot(**plotting_parameters[ind])

            # Add annotations
            annotator = Annotator(ax, pairs, **plotting_parameters[ind])
            (annotator
             .configure(test=None, text_format='star')
             .set_pvalues(pvalues=p_values[ind])
             .annotate())
            plt.title(f"INPSR Results {condition}")
            plt.savefig(f"{figure_path}SNR_{condition}")

        plt.show()
