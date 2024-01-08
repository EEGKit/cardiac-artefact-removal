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
    run_anova = False
    run_posthoc = True
    plot_graph = True

    # Read in results files and format in a pandas dataframe
    # Set file locations
    fname = 'snr.h5'
    prep_path = '/data/pt_02569/tmp_data/prepared_py/'
    pca_path = '/data/pt_02569/tmp_data/ecg_rm_py/'
    ica_path = '/data/pt_02569/tmp_data/baseline_ica_py/'
    ssp_path = '/data/p_02569/SSP/'
    figure_path = '/data/p_02569/StatsGraphs_Dataset1/'
    os.makedirs(figure_path, exist_ok=True)

    ################################# Make Dataframe ###################################
    file_paths = [prep_path, pca_path, ica_path, ssp_path]
    names = ['Prep', 'PCA', 'ICA', 'SSP']
    names_indf = ['Prep', 'PCA', 'ICA', 'SSP_5', 'SSP_6']
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

    df_med.to_csv(path_or_buf='/data/pt_02569/Dataframe.csv')
    exit()
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
    target = 'Prep'
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

            # Plot with seaborn
            ax = sns.boxplot(**plotting_parameters[ind])

            # Add annotations
            annotator = Annotator(ax, pairs, **plotting_parameters[ind])
            (annotator
             .configure(test=None, text_format='star')
             .set_pvalues(pvalues=p_values[ind])
             .annotate())
            plt.title(f"SNR Results {condition}")
            plt.savefig(f"{figure_path}SNR_{condition}")

        plt.show()

    ############################# Graveyard ###############################
    # # ONE-WAY ANOVA
    # data = [df_med[col].dropna() for col in df_med]  # Needs to ignore nan values
    # fvalue, pvalue = stats.f_oneway(*data)
    # # 23.443565187413647 1.7097586806094394e-18
    #
    # data = [df_tib[col].dropna() for col in df_tib]  # Needs to ignore nan values
    # fvalue, pvalue = stats.f_oneway(*data)
    # # 19.346673058152767 9.534707891876217e-16

    # # Get as an R-like output - gives same numbers as F and P values above
    # # Formatting with bioinfokit
    # res_m = stat()
    # res_m.anova_stat(df=df_med_melt, res_var='value', anova_model='value ~ C(method)')
    # # print('Median')
    # # print(res.anova_summary)
    #
    # res_t = stat()
    # res_t.anova_stat(df=df_tib_melt, res_var='value', anova_model='value ~ C(method)')
    # # print('Tibial')
    # # print(res.anova_summary)

    # # TUKEY HSD with pingouin
    # stats_med_tukey = pingouin.pairwise_tukey(data=df_med_melt, dv='value', between='method', effsize='hedges')
    # # print('Median Tukey')
    # # print(stats_med_tukey[['A', 'B', 'p-tukey']])
    # stats_tib_tukey = pingouin.pairwise_tukey(data=df_tib_melt, dv='value', between='method', effsize='hedges')
    # # print('Tibial Tukey')
    # # print(stats_tib_tukey[['A', 'B', 'p-tukey']])
    #
    # # PAIRWISE T-TESTS with bonferroni correction
    # stats_med_t = pingouin.pairwise_tests(data=df_med_melt, dv='value', between='method', within=None,
    #                                       subject=None, parametric=True, marginal=True, alpha=0.05,
    #                                       alternative='two-sided', padjust='bonf', effsize='hedges',
    #                                       correction='auto', nan_policy='listwise', return_desc=True,
    #                                       interaction=False, within_first=True)
    # # print('Median Pairwise T-Test')
    # # print(stats_med_t[['A', 'B', 'p-unc', 'p-corr']])
    # stats_tib_t = pingouin.pairwise_tests(data=df_tib_melt, dv='value', between='method', within=None,
    #                                       subject=None, parametric=True, marginal=True, alpha=0.05,
    #                                       alternative='two-sided', padjust='bonf', effsize='hedges',
    #                                       correction='auto', nan_policy='listwise', return_desc=True,
    #                                       interaction=False, within_first=True)
    # print('Tibial Pairwise T-Test')
    # print(stats_tib_t[['A', 'B', 'p-unc', 'p-corr']])

    # # generate a boxplot to see the data distribution by treatments.
    # plt.figure()
    # ax1 = sns.boxplot(x='method', y='value', data=df_med_melt, color='darkblue')
    # ax1 = sns.swarmplot(x="method", y="value", data=df_med_melt, color='#7d0013')
    # plt.title('Median Stimulation SNR')
    #
    # plt.figure()
    # ax2 = sns.boxplot(x='method', y='value', data=df_tib_melt, color='deepskyblue')
    # ax2 = sns.swarmplot(x="method", y="value", data=df_tib_melt, color='#7d0013')
    # plt.title('Tibial Stimulation SNR')
    # plt.show()

    # # Test assumptions of ANOVA GRAPHICALLYY
    # import statsmodels.api as sm
    #
    # # res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
    # sm.qqplot(res_m.anova_std_residuals, line='45')
    # plt.xlabel("Theoretical Quantiles")
    # plt.ylabel("Standardized Residuals")
    # plt.title('Median Stimulation')
    # plt.show()
    #
    # sm.qqplot(res_t.anova_std_residuals, line='45')
    # plt.xlabel("Theoretical Quantiles")
    # plt.ylabel("Standardized Residuals")
    # plt.title('Tibial Stimulation')
    # plt.show()
    #
    # # histogram
    # plt.hist(res_m.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
    # plt.xlabel("Residuals")
    # plt.ylabel('Frequency')
    # plt.title('Median Stimulation')
    # plt.show()
    #
    # plt.hist(res_t.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
    # plt.xlabel("Residuals")
    # plt.ylabel('Frequency')
    # plt.title('Tibial Stimulation')
    # plt.show()
    # exit()

    # # Moved away from bioinfokit to using pingouin instead
    # # Tukey HSD test to see which are different
    # res = stat()
    # res.tukey_hsd(df=df_med_melt, res_var='value', xfac_var='method', anova_model='value ~ C(method)')
    # print('Median')
    # print(res.tukey_summary)
    #
    # res = stat()
    # res.tukey_hsd(df=df_tib_melt, res_var='value', xfac_var='method', anova_model='value ~ C(method)')
    # # print('Tibial')
    # # print(res.tukey_summary)
    # # Note: p-value 0.001 from tukey_hsd output should be interpreted as <=0.001