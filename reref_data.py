import mne
from get_conditioninfo import *
from scipy.io import loadmat
import os

def rereference_data(raw, ch_name):
    if ch_name in raw.ch_names:
        raw_ref = raw.copy().set_eeg_reference(ref_channels=[ch_name])
    else:
        raw_ref = raw.copy().set_eeg_reference(ref_channels='average')

    return raw_ref