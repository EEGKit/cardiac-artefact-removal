# Get the standard error of the mean across subjects for snr, inpsr, ri and coeff of variation
# Make sure you run code to get updated sem values if anything in code has been changed since previous run


import pandas as pd
import matplotlib.pyplot as plt
import h5py
import colorsys
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    fname = 'variance_controlptp.h5'  # variance or variance_controlptp
    keywords = ['var_med', 'var_tib']
    for cca_flag in [True, False]:
        names = ['Prepared', 'PCA', 'SSP']
        if cca_flag:
            prep_path = '/data/pt_02569/tmp_data/prepared_py_cca/'
            pca_path = '/data/pt_02569/tmp_data/ecg_rm_py_cca/'
            ssp_path = '/data/pt_02569/tmp_data/ssp_py_cca/'

        else:
            prep_path = '/data/pt_02569/tmp_data/prepared_py/'
            pca_path = '/data/pt_02569/tmp_data/ecg_rm_py/'
            ssp_path = '/data/pt_02569/tmp_data/ssp_py/'

        file_paths = [prep_path, pca_path, ssp_path]
        count = 0
        for file_path in file_paths:
            with h5py.File(file_path + fname, "r") as infile:
                if file_path == prep_path:
                    snr_med = infile[keywords[0]][()].reshape(-1)
                    data_med = {'Prep': snr_med}

                    snr_tib = infile[keywords[1]][()].reshape(-1)
                    data_tib = {'Prep': snr_tib}
                    if cca_flag:
                        df_med_a = pd.DataFrame(data_med, index=np.arange(1, 37))
                        df_tib_a = pd.DataFrame(data_tib, index=np.arange(1, 37))
                    else:
                        df_med_b = pd.DataFrame(data_med, index=np.arange(1, 37))
                        df_tib_b = pd.DataFrame(data_tib, index=np.arange(1, 37))

                elif file_path == ssp_path:
                    # These have shape (n_subjects, n_projectors)
                    snr_med = infile[keywords[0]][()]
                    snr_tib = infile[keywords[1]][()]

                    if cca_flag:
                        df_med_a[f'{names[count]}_5'] = snr_med[:, 0]
                        df_med_a[f'{names[count]}_6'] = snr_med[:, 1]
                        df_tib_a[f'{names[count]}_5'] = snr_tib[:, 0]
                        df_tib_a[f'{names[count]}_6'] = snr_tib[:, 1]
                    else:
                        df_med_b[f'{names[count]}_5'] = snr_med[:, 0]
                        df_med_b[f'{names[count]}_6'] = snr_med[:, 1]
                        df_tib_b[f'{names[count]}_5'] = snr_tib[:, 0]
                        df_tib_b[f'{names[count]}_6'] = snr_tib[:, 1]

                else:
                    # Get the data
                    snr_med = infile[keywords[0]][()].reshape(-1)

                    snr_tib = infile[keywords[1]][()].reshape(-1)
                    if cca_flag:
                        df_med_a[names[count]] = snr_med
                        df_tib_a[names[count]] = snr_tib
                    else:
                        df_med_b[names[count]] = snr_med
                        df_tib_b[names[count]] = snr_tib

                count += 1

    # ###################### Get mean and sem of columns ####################
    # print('AFTER')
    # print('Median Means')
    # print(df_med_a.mean())
    # print('Median Standard Error')
    # print(df_med_a.sem())
    # print('Tibial Means')
    # print(df_tib_a.mean())
    # print('Tibial Standard Error')
    # print(df_tib_a.sem())
    #
    # print('\n')
    # print('BEFORE')
    # print('Median Means')
    # print(df_med_b.mean())
    # print('Median Standard Error')
    # print(df_med_b.sem())
    # print('Tibial Means')
    # print(df_tib_b.mean())
    # print('Tibial Standard Error')
    # print(df_tib_b.sem())

    for df in [df_med_a, df_tib_a, df_med_b, df_tib_b]:
        df.drop('SSP_5', axis=1, inplace=True)
        df.rename({'Prep':'Uncleaned', 'PCA': 'PCA_OBS', 'SSP_6':'SSP'}, axis=1, inplace=True)

    # Get dataframe in form easy to plot
    # Means
    df_mean = pd.DataFrame(columns=['Method', 'Median_b', 'Median_a', 'Tibial_b', 'Tibial_a'])
    df_mean['Method'] = ['Uncleaned', 'PCA-OBS', 'SSP']
    df_sem = pd.DataFrame(columns=['Method', 'Median_b', 'Median_a', 'Tibial_b', 'Tibial_a'])
    df_sem['Method'] = ['Uncleaned', 'PCA-OBS', 'SSP']
    for name, df_check in zip(['Median_b', 'Median_a', 'Tibial_b', 'Tibial_a'],
                               [df_med_b, df_med_a, df_tib_b, df_tib_a]):
        df_mean[name] = df_check.mean().values
        df_sem[name] = df_check.sem().values
    df_mean.set_index('Method', inplace=True)
    df_sem.set_index('Method', inplace=True)

    colours = ['seagreen', 'lightgreen']

    for condition in ['Median', 'Tibial']:
        # CCA coeff of variation plots
        methods_toplot = ['Uncleaned', 'PCA-OBS', 'SSP']
        columns = [f'{condition}_b', f'{condition}_a']
        df_cropped_mean = df_mean.loc[methods_toplot, columns]
        df_cropped_sem = df_sem.loc[methods_toplot, columns]
        fig, ax = plt.subplots(1, 1)
        ax.set_xticklabels(methods_toplot)
        ax.set_ylabel('Coefficient of Variation (AU)')
        ax.set_title(f'Variability in Single Trial Amplitudes \n{condition} Nerve Stimulation')
        df_cropped_mean.plot.bar(ax=ax, rot=15, color=colours, sharex=True, legend=0, yerr=df_cropped_sem, capsize=4)
        plt.legend(['Before CCA', 'After CCA'], loc='upper right')
        plt.tight_layout()

        if fname == 'variance.h5':
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Var_{condition}_D1.png')
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Var_{condition}_D1.pdf', bbox_inches='tight', format="pdf")
        else:
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Var_{condition}_D1_controlptp.png')
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Var_{condition}_D1_controlptp.pdf', bbox_inches='tight', format="pdf")

    # plt.show()
