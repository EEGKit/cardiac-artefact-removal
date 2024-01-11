# Generate rainbow plots of SEP SNR

import pandas as pd
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':

    # Plotting fewer for purposes of TAC meeting
    colours = ['darkblue', 'deepskyblue']  # Colour 1 for median, Colour 2 for tibial
    methods_toplot_prep = ['Prepared', 'PCA', 'ICA', 'SSP6']
    names_withprep = ['Prepared', 'PCA', 'ICA', 'SSP']
    names_noprep = ['PCA', 'ICA', 'SSP']
    methods_toplot = ['PCA', 'ICA', 'SSP6']

    ######################### SEP SNR #########################
    keywords = ['snr_med', 'snr_tib']
    input_paths = ["/data/pt_02569/tmp_data/prepared_py/",
                   "/data/pt_02569/tmp_data/ecg_rm_py/",
                   "/data/pt_02569/tmp_data/baseline_ica_py/",
                   "/data/p_02569/SSP/"]

    names = ['Prepared', 'PCA', 'ICA', 'SSP']
    i = 0
    snr_list_med = {}
    snr_list_tib = {}
    for input_path in input_paths:
        name = names[i]
        fn = f"{input_path}snr.h5"

        # All have shape (36, 1) bar SSP which is (36, n_proj)
        with h5py.File(fn, "r") as infile:
            if name == 'SSP':
                snr_med = infile[keywords[0]][()]
                snr_tib = infile[keywords[1]][()]
                for i, (data_med, data_tib) in enumerate(zip(snr_med.T, snr_tib.T)):
                    snr_list_med[f'{name}_{i+1}'] = data_med
                    snr_list_tib[f'{name}_{i+1}'] = data_tib
            else:
                # Get the data
                snr_med = infile[keywords[0]][()].reshape(-1)
                snr_tib = infile[keywords[1]][()].reshape(-1)
                snr_list_med[name] = snr_med
                snr_list_tib[name] = snr_tib

        i += 1

    df_med = pd.DataFrame(snr_list_med)
    df_med.rename(columns={'Prepared': 'Uncleaned', 'SSP_6': 'SSP', 'PCA': 'PCA-OBS'}, inplace=True)
    df_med = df_med[['Uncleaned', 'PCA-OBS', 'ICA', 'SSP']]
    df_tib = pd.DataFrame(snr_list_tib)
    df_tib.rename(columns={'Prepared': 'Uncleaned', 'SSP_6': 'SSP', 'PCA': 'PCA-OBS'}, inplace=True)
    df_tib = df_tib[['Uncleaned', 'PCA-OBS', 'ICA', 'SSP']]

    df_med_long = df_med.melt(var_name='Method', value_name='SNR (A.U.)')
    df_tib_long = df_tib.melt(var_name='Method', value_name='SNR (A.U.)')

    dy = "SNR (A.U.)"
    dx = "Method"
    ort = "v"
    pal = sns.color_palette(n_colors=4)
    i = 0
    conditons = ['median', 'tibial']
    for df in [df_med_long, df_tib_long]:
        cond_name = conditons[i]
        i += 1
        f, ax = plt.subplots(figsize=(5, 8))
        ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.2, cut=0.,
                                scale="area", width=.6, inner=None, orient=ort,
                                linewidth=0.0)
        ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                           size=3, jitter=1, zorder=0, orient=ort)
        # ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10,
        #                  showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
        #                  showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
        #                  saturation=1, orient=ort)
        # ax = ax.set_ylim[(0, 25)]
        plt.ylim([0, 25])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if cond_name == 'median':
            plt.title("SEP Signal-to-Noise Ratio\n Median Nerve Stimulation - Cervical Cord")
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Rainbow_{cond_name}')
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Rainbow_{cond_name}.pdf', bbox_inches='tight',
                        format="pdf")
        else:
            plt.title("SEP Signal-to-Noise Ratio\n Tibial Nerve Stimulation - Lumbar Cord")
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Rainbow_{cond_name}')
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Rainbow_{cond_name}.pdf', bbox_inches='tight',
                        format="pdf")
    plt.show()
