#################################################################################
# INPSR Relevant Channels
#################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import colorsys
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    colours = ['darkblue', 'deepskyblue']  # Colour 1 for median, Colour 2 for tibial
    fname = 'snr.h5'
    prep_path = '/data/pt_02569/tmp_data/prepared_py/'
    pca_path = '/data/pt_02569/tmp_data/ecg_rm_py/'
    ica_path = '/data/pt_02569/tmp_data/baseline_ica_py/'
    ssp_path = '/data/pt_02569/tmp_data/ssp_py/'

    ################################# Make Dataframe ###################################
    file_paths = [prep_path, pca_path, ica_path, ssp_path]
    names = ['Prep', 'PCA', 'ICA', 'SSP']
    names_indf = ['Prep', 'PCA', 'ICA', 'SSP_5', 'SSP_6']
    # Pull each subjects value out
    keywords = ['snr_med', 'snr_tib']
    count = 0
    for file_path in file_paths:
        with h5py.File(file_path + fname, "r") as infile:
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
                df_med[f'{names[count]}_5'] = snr_med[:, 4]
                df_med[f'{names[count]}_6'] = snr_med[:, 5]

                snr_tib = infile[keywords[1]][()]
                df_tib[f'{names[count]}_5'] = snr_tib[:, 4]
                df_tib[f'{names[count]}_6'] = snr_tib[:, 5]

            else:
                # Get the data
                snr_med = infile[keywords[0]][()].reshape(-1)
                df_med[names[count]] = snr_med

                snr_tib = infile[keywords[1]][()].reshape(-1)
                df_tib[names[count]] = snr_tib

            count += 1

    # Drop SSP_5
    df_med.drop('SSP_5', axis=1, inplace=True)
    # Drop SSP_5
    df_tib.drop('SSP_5', axis=1, inplace=True)
    ###################### Get mean and sem of columns ####################
    print('Median Means')
    print(df_med.mean())
    print('Median Standard Error')
    print(df_med.sem())
    print('Tibial Means')
    print(df_tib.mean())
    print('Tibial Standard Error')
    print(df_tib.sem())

    # Plot
    for name, df in zip(['median', 'tibial'], [df_med, df_tib]):
        df.rename({'PCA': 'PCA_OBS'}, axis=1, inplace=True)
        df.rename({'SSP_6': 'SSP'}, axis=1, inplace=True)
        plt.figure()
        if name == 'median':
            col = colours[0]
            plt.title('SEP Signal-to-Noise Ratio \nMedian Nerve Stimulation - Cervical Cord')
        else:
            col = colours[1]
            plt.title('SEP Signal-to-Noise Ratio \nTibial Nerve Stimulation - Lumbar Cord')
        df.mean().plot.bar(rot=15, color=col, yerr=df.sem(), capsize=4)
        plt.ylabel('SNR (AU)')
        plt.ylim([0, 16])
        plt.tight_layout()
        plt.savefig(f'/data/pt_02569/ResultsComparison/Images/SEP_SNR_Separated_Separated_{name.capitalize()}_D1.png')
        plt.savefig(f'/data/pt_02569/ResultsComparison/Images/SEP_SNR_Separated_Separated_{name.capitalize()}_D1.pdf', bbox_inches='tight', format="pdf")

    plt.show()
