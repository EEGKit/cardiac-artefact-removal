################################################################################################
# Residual Intensity Relevant Channels
################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import colorsys
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    methods_toplot = ['PCA', 'ICA', 'SSP6']
    colours = ['darkblue', 'deepskyblue']  # Colour 1 for median, Colour 2 for tibial

    fname = 'res.h5'
    prep_path = '/data/pt_02569/tmp_data/prepared_py/'
    pca_path = '/data/pt_02569/tmp_data/ecg_rm_py/'
    ica_path = '/data/pt_02569/tmp_data/baseline_ica_py/'
    ssp_path = '/data/pt_02569/tmp_data/ssp_py/'

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

    # Set up dataframe for stats
    file_paths = [prep_path, pca_path, ica_path, ssp_path]
    names = ['Prep', 'PCA', 'ICA', 'SSP_5', 'SSP_6']  # To make dataframe
    names_indf = ['PCA', 'ICA', 'SSP_5', 'SSP_6']  # To access columns in df
    # Pull each subjects value out
    keywords = ['res_med', 'res_tib']
    count = 0
    for file_path in file_paths:
        if file_path == ssp_path:
            fname = 'res_5.h5'

        # Need the prep values to do all the divisions thereafter
        with h5py.File(file_path + fname, "r") as infile:
            if file_path == prep_path:  # Just extract values from file to use later
                val_prep_med = infile[keywords[0]][()]
                val_prep_tib = infile[keywords[1]][()]

            elif file_path == pca_path:  # Start the dataframe
                # Get the data
                val_med = infile[keywords[0]][()]
                res_med_current = (np.mean(val_med[:, median_pos] / val_prep_med[:, median_pos],
                                           axis=1)) * 100
                data_med = {'PCA': res_med_current}
                df_med = pd.DataFrame(data_med, index=np.arange(1, 37))

                val_tib = infile[keywords[1]][()]
                res_tib_current = (np.mean(val_tib[:, tibial_pos] / val_prep_tib[:, tibial_pos],
                                           axis=1)) * 100
                data_tib = {'PCA': res_tib_current}
                df_tib = pd.DataFrame(data_tib, index=np.arange(1, 37))

            else:
                # Get the data
                val_med = infile[keywords[0]][()]
                res_med_current = (np.mean(val_med[:, median_pos] / val_prep_med[:, median_pos],
                                           axis=1)) * 100
                df_med[names[count]] = res_med_current

                val_tib = infile[keywords[1]][()]
                res_tib_current = (np.mean(val_tib[:, tibial_pos] / val_prep_tib[:, tibial_pos],
                                           axis=1)) * 100
                df_tib[names[count]] = res_tib_current

            count += 1
    # Then just add SSP 6 to the last column
    fname = 'res_6.h5'
    with h5py.File(ssp_path + fname, "r") as infile:
        val_med = infile[keywords[0]][()]
        res_med_current = (np.mean(val_med[:, median_pos] / val_prep_med[:, median_pos],
                                   axis=1)) * 100
        df_med['SSP_6'] = res_med_current

        val_tib = infile[keywords[1]][()]
        res_tib_current = (np.mean(val_tib[:, tibial_pos] / val_prep_tib[:, tibial_pos],
                                   axis=1)) * 100
        df_tib['SSP_6'] = res_tib_current

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
            plt.title('Residual Intensity of the Cardiac Artefact \nMedian Nerve Stimulation - Cervical Cord')
        else:
            col = colours[1]
            plt.title('Residual Intensity of the Cardiac Artefact \nTibial Nerve Stimulation - Lumbar Cord')
        df.mean().plot.bar(rot=15, color=col, yerr=df.sem(), capsize=4)
        plt.ylabel('Residual Intensity (%)')
        plt.ylim([0, 1.7])
        plt.tight_layout()
        plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Intensity_Separated_{name.capitalize()}_D1.png')
        plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Intensity_Separated_{name.capitalize()}_D1.pdf', bbox_inches='tight', format="pdf")

    plt.show()