#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from collections import Counter
import numpy as np
import scipy.io as sio
from scipy.signal import detrend
import mne
mne.set_log_level(verbose='WARNING')
from mne.filter import filter_data, notch_filter
#from extract_features_parallel import *


seg_mask_explanation = [
    'normal',
    'around sleep stage change point',
    'marked by multiple sleep stages in the same epoch',
    'NaN in sleep stage',
    'NaN in EEG',
    'overly high/low amplitude',
    'flat signal',
    'spurious spectrum',
    'muscle artifact',
    'NaN in feature']

    
def segment_EEG(EEG, epoch_time, Fs, newFs, NW, amplitude_thres=500, notch_freq=None, bandpass_freq=None, start_end_remove_window_num=0, to_remove_mean=False, n_jobs=1, subject_file_name=''):#
    """Segment EEG signals.
    """
    std_thres1 = 0.02
    std_thres2 = 0.1
    flat_seconds = 2
    
    if to_remove_mean:
        EEG = EEG - np.mean(EEG,axis=1, keepdims=True)
    epoch_size = int(round(epoch_time*Fs))
    flat_length = int(round(flat_seconds*Fs))
    
    ## filtering
    
    EEG = notch_filter(EEG, Fs, notch_freq, n_jobs=-1, verbose='error')  # (#window, #ch, epoch_size+2padding)
    EEG = filter_data(detrend(EEG, axis=1), Fs, bandpass_freq[0], bandpass_freq[1], n_jobs=-1, verbose='error')
    
    ## segment
    
    start_ids = np.arange(0, EEG.shape[1]-epoch_size+1, epoch_size)
    if start_end_remove_window_num>0:
        start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
    seg_masks = [seg_mask_explanation[0]]*len(start_ids)
    EEG_segs = EEG[:, list(map(lambda x:np.arange(x,x+epoch_size), start_ids))].transpose(1,0,2)  # (#window, #ch, epoch_size+2padding)
    
    ## resampling
    
    mne_epochs = mne.EpochsArray(detrend(EEG_segs, axis=2), mne.create_info(ch_names=list(map(str, range(EEG_segs.shape[1]))), sfreq=Fs, ch_types='eeg'), verbose=False)
    if newFs!=Fs:
        Fs = newFs
        mne_epochs.resample(Fs, n_jobs=n_jobs)
        EEG_segs = mne_epochs.get_data()
        epoch_size = int(round(epoch_time*Fs))
        flat_length = int(round(flat_seconds*Fs))
    
    ## calculate spectrogram
    
    BW = NW*2./epoch_time
    specs, freq = mne.time_frequency.psd_multitaper(mne_epochs, fmin=bandpass_freq[0], fmax=bandpass_freq[1], adaptive=False, low_bias=False, n_jobs=n_jobs, verbose='ERROR', bandwidth=BW, normalization='full')
    
    ## mark artifacts
    
    # nan in signal
    nan2d = np.any(np.isnan(EEG_segs), axis=2)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    for i in nan1d:
        seg_masks[i] = seg_mask_explanation[4]
            
    # flat signal
    short_segs = EEG_segs.reshape(EEG_segs.shape[0], EEG_segs.shape[1], EEG_segs.shape[2]//flat_length, flat_length)
    flat2d = np.any(detrend(short_segs, axis=3).std(axis=3)<=std_thres1, axis=2)
    flat2d = np.logical_or(flat2d, np.std(EEG_segs,axis=2)<=std_thres2)
    flat1d = np.where(np.any(flat2d, axis=1))[0]
    for i in flat1d:
        seg_masks[i] = seg_mask_explanation[6]
            
    # big amplitude
    amplitude_large2d = np.any(np.abs(EEG_segs)>amplitude_thres, axis=2)
    amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0]
    for i in amplitude_large1d:
        seg_masks[i] = seg_mask_explanation[5]

    return EEG_segs, start_ids, seg_masks, specs, freq



