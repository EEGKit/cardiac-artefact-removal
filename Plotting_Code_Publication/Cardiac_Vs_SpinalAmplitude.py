import os
import numpy as np
import mne
from scipy.io import loadmat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
from Metrics.SNR_functions import evoked_from_raw
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    calculate = False
    cond_names = ['median', 'tibial']
    save_path = '/data/p_02569/CardiacVsSpinalAmplitudes_D1/'
    os.makedirs(save_path, exist_ok=True)

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000
    iv_baseline_ecg = [-300 / 1000, -200 / 1000]
    iv_epoch_ecg = [-300 / 1000, 400 / 1000]

    subjects = np.arange(1, 37)
    sampling_rate = 1000

    if calculate:
        amp_esg_med = []
        amp_ecg_med = []
        amp_esg_tib = []
        amp_ecg_tib = []
        for subj in subjects:
            subject_id = f'sub-{str(subj).zfill(3)}'
            potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
            fname_pot = 'potential_latency.mat'
            matdata = loadmat(potential_path + fname_pot)

            for cond_name in cond_names:

                if cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channel = ['SC6']
                    sep_latency = matdata['med_potlatency'][0][0]/1000

                elif cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channel = ['L1']
                    sep_latency = matdata['tib_potlatency'][0][0]/1000

                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr1000_{cond_name}_withqrs.fif", preload=True)

                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names),
                           method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked = evoked.pick_channels(channel)

                evoked_ecg = evoked_from_raw(raw, iv_epoch_ecg, iv_baseline_ecg, 'qrs', False)
                evoked_ecg = evoked_ecg.pick_channels(channel)

                # plt.figure()
                # plt.plot(evoked.times, evoked.data.reshape(-1))
                # plt.figure()
                # plt.plot(evoked_ecg.times, evoked_ecg.data.reshape(-1))

                ch_name, lat, amp = evoked.get_peak(ch_type=None, tmin=sep_latency-2/1000,
                                                    tmax=sep_latency+2/1000, mode='neg',
                                                    time_as_index=False, return_amplitude=True)
                if cond_name == 'tibial':
                    mode = 'pos'
                else:
                    mode = 'neg'
                ch_ecg, lat_ecg, amp_h = evoked_ecg.get_peak(ch_type=None, tmin=-20/1000,
                                                             tmax=20/1000, mode=mode,
                                                             time_as_index=False, return_amplitude=True)

                if cond_name == 'median':
                    amp_esg_med.append(amp*10**6)
                    amp_ecg_med.append(amp_h*10**6)
                else:
                    amp_esg_tib.append(amp*10**6)
                    amp_ecg_tib.append(amp_h*10**6)

        df = pd.DataFrame(
            {'Subjects': subjects,
             'ESG_Median': amp_esg_med,
             'ECG_Median': amp_ecg_med,
             'ESG_Tibial': amp_esg_tib,
             'ECG_Tibial': amp_ecg_tib
             })

        df.to_excel(save_path + 'Amplitudes.xlsx')

    else:
        df = pd.read_excel(save_path + 'Amplitudes.xlsx')
        df.set_index('Subjects')

    pd.set_option('expand_frame_repr', False)
    print(df.describe())
    df_med = df[['ESG_Median', 'ECG_Median']]
    df_tib = df[['ESG_Tibial', 'ECG_Tibial']]
    df_med_long = df_med.melt(var_name='Signal Type', value_name='Amplitude')
    df_tib_long = df_tib.melt(var_name='Signal Type', value_name='Amplitude')

    dy = "Amplitude"
    dx = "Signal Type"
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
        ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10,
                         showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                         showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                         saturation=1, orient=ort)
        # plt.ylim([0, 25])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.show()
