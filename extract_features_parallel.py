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


import pdb
import subprocess
import sys
import numpy as np
import scipy.stats as stats
from scipy.signal import detrend
from joblib import Parallel, delayed
import nitime.algorithms as tsa
#from pyeeg import samp_entropy
#import pyrem.univariate as pu
from multitaper_spectrogram import *
from bandpower import *
SAMPEN_PATH = '/data/sleep_staging/sleep_staging_for_Dhina/physionet.org/files/sampen/1.0.0/c/sampen'  # https://www.physionet.org/physiotools/sampen/


#pxx_mts = []
#freqs = []
#workers_done = 0

#NW = 2
#total_freq_range = [0.5,20]  # [Hz]
#window_length = 2  # [s]
#window_step = 1  # [s]
band_names = ['delta','theta','alpha','sigma']
band_freq = [[0.5,4],[4,8],[8,12],[12,20]]  # [Hz]
band_num = len(band_freq)
combined_channel_names = ['F','C','O']
combined_channel_num = len(combined_channel_names)

#def compute_pxx_mt(EEG_seg, Fs, NW, window_length, window_step, segi, seg_num):
#    # get multitaper spectrogram
#    pxx_mt, freq = multitaper_spectrogram(EEG_seg, Fs, NW, window_length, window_step)
#
#    # take the average between left and right channels  ##########################THIS DEPENDS ON THE ACTUAL CHANNEL ORDERING!!!!!!
#    pxx_mt[:,:,:-1] = (pxx_mt[:,:,:-1]+pxx_mt[:,:,1:])/2.0
#    pxx_mt = np.delete(pxx_mt,[1,3,5],axis=2)
#
#    return pxx_mt, freq, segi, seg_num

#def record_pxx_mt(result):
#    global workers_done
#    pxx_mts[result[2]] = result[0]
#    freqs[result[2]] = result[1]
#    workers_done = workers_done + 1
#    if workers_done%10==0:
#        print('%d/%d segments competed.'%(workers_done,result[3]))


def compute_features_each_seg(eeg_seg, seg_size, channel_num, band_num, NW, Fs, freq, band_freq, total_freq_range, total_freq_id, window_length, window_step, dpss=None, eigvals=None, return_spec_only=False):
    # psd estimation, size=(window_num, freq_point_num, channel_num)
    # frequencies, size=(freq_point_num,)
    spec_mt = multitaper_spectrogram(eeg_seg, Fs, NW, window_length, window_step, dpss=dpss, eigvals=eigvals)
    #total_findex = [i for i in range(len(freq)) if total_freq_range[0]<=freq[i]<total_freq_range[1]]
    
    if return_spec_only:
        if total_freq_id is None:
            return spec_mt[0] #############1 sub-epoch
        else:
            return spec_mt[0,total_freq_id,:] #############1 sub-epoch
        
    spec_mt = (spec_mt[:,:,[0,2,4]]+spec_mt[:,:,[1,3,5]])/2.0
    combined_channel_num = spec_mt.shape[2]

    # relative band power using multitaper
    bandpower_mt, band_findex = bandpower(spec_mt, freq, band_freq, total_freq_range=total_freq_range, relative=False)
    #bandpower_mt = [bandpower_mt[i].squeeze() for i in range(band_num)]

    f1 = np.abs(np.diff(eeg_seg,axis=1)).sum(axis=1)*1.0/seg_size  # line length
    f2 = stats.kurtosis(eeg_seg,axis=1,nan_policy='propagate')  # kurtosis
    # sample entropy
    f3 = []
    for ci in range(channel_num):
        #Bruce, E. N., Bruce, M. C., & Vennelaganti, S. (2009).
        #Sample entropy tracks changes in EEG power spectrum with sleep state and aging. Journal of clinical neurophysiology, 26(4), 257.
        #f3.append(pu.samp_entropy(eeg_seg[ci],2,0.2,relative_r=True))  
        sp = subprocess.Popen([SAMPEN_PATH,'-m','2','-r','0.2','-n'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)#       
        f3.append(float(sp.communicate(input=np.array2string(eeg_seg[ci], suppress_small=False, separator='\n')[1:-1].encode())[0].decode().split('=')[-1]))

    f4 = [];f5 = [];f6 = [];f7 = [];f9 = []#;f8 = []
    for bi in range(band_num):
        if bi!=band_num-1: # no need for sigma band
            f4.extend(np.percentile(bandpower_mt[bi],95,axis=0))
            f5.extend(bandpower_mt[bi].min(axis=0))
            f6.extend(bandpower_mt[bi].mean(axis=0))
            f7.extend(bandpower_mt[bi].std(axis=0))

        spec_flatten = spec_mt[:,band_findex[bi],:].reshape(spec_mt.shape[0]*len(band_findex[bi]),spec_mt.shape[2])
        ###############f8.extend(stats.skew(spec_flatten, axis=0, nan_policy='propagate'))  # skewness
        f9.extend(stats.kurtosis(spec_flatten, axis=0, nan_policy='propagate'))  # kurtosis
        # decide bin size for Shannon entropy
        # Izenman, AJ. (1991). Recent developments in nonparametric density estimation. J AM STAT ASSOC, 86(413), 205-224
        #q95,q75,q25,q5 = np.percentile(spec_flatten, [95,75,25,5], axis=0)
        #bin_num = np.array((q95-q5)/(2.0*(q75-q25)*np.power(spec_flatten.shape[0],-0.33333)),dtype=int)
        #for chi in range(combined_channel_num):
        #    f10.append(stats.entropy(np.histogram(spec_flatten[:,chi], bins=bin_num[chi])[0]))  # Shannon entropy of the spectrogram in each band

    #band_names = ['delta','theta','alpha','sigma']
    f10 = []
    delta_theta = bandpower_mt[0]/(bandpower_mt[1]+1)
    f10.extend(np.percentile(delta_theta,95,axis=0))
    f10.extend(np.min(delta_theta,axis=0))
    f10.extend(np.mean(delta_theta,axis=0))
    f10.extend(np.std(delta_theta,axis=0))
    f11 = []
    delta_alpha = bandpower_mt[0]/(bandpower_mt[2]+1)
    f11.extend(np.percentile(delta_alpha,95,axis=0))
    f11.extend(np.min(delta_alpha,axis=0))
    f11.extend(np.mean(delta_alpha,axis=0))
    f11.extend(np.std(delta_alpha,axis=0))
    f12 = []
    theta_alpha = bandpower_mt[1]/(bandpower_mt[2]+1)
    f12.extend(np.percentile(theta_alpha,95,axis=0))
    f12.extend(np.min(theta_alpha,axis=0))
    f12.extend(np.mean(theta_alpha,axis=0))
    f12.extend(np.std(theta_alpha,axis=0))

    return np.r_[f1,f2,f3,f4,f5,f6,f7,f9,f10,f11,f12]#f8


def extract_features(EEG_segs, channel_names, combined_channel_names, Fs, NW, total_freq_range, sub_window_time, sub_window_step, seg_start_ids, return_feature_names=False, n_jobs=-1, verbose=True):
    """Extract features from EEG segments.

    Arguments:
    EEG_segs -- a list of EEG segments in numpy.ndarray type, size=(sample_point, channel_num)
    channel_names -- a list of channel names for each column of EEG_segs
    ##combined_channel_names -- a list of combined column_channels_names, for example from 'F3M2' and 'F4M1' to 'F'
    Fs -- sampling frequency in Hz

    Keyword arguments:
    process_num -- default None, number of parallel processes, if None, equals to 4x #CPU.

    Outputs:
    features from each segment in numpy.ndarray type, size=(seg_num, feature_num)
    a list of names of each feature
    psd estimation, size=(window_num, freq_point_num, channel_num), or a list of them for each band
    frequencies, size=(freq_point_num,), or a list of them for each band
    """

    #if type(EEG_segs)!=list:
    #    raise TypeError('EEG segments should be list of numpy.ndarray, with size=(sample_point, channel_num).')

    seg_num, channel_num, window_size = EEG_segs.shape
    if seg_num <= 0:
        return []
    
    sub_window_size = int(round(sub_window_time*Fs))
    sub_step_size = int(round(sub_window_step*Fs))
    dpss, eigvals = tsa.dpss_windows(sub_window_size,NW,2*NW)
    nfft = max(1<<(sub_window_size-1).bit_length(), sub_window_size)
    freq = np.arange(0, Fs, Fs*1.0/nfft)[:nfft//2+1]
    total_freq_id = np.where(np.logical_and(freq>=total_freq_range[0], freq<total_freq_range[1]))[0]
    
    old_threshold = np.get_printoptions()['threshold']
    np.set_printoptions(threshold=sys.maxsize)#np.nan)
    
    features = Parallel(n_jobs=n_jobs,verbose=verbose,backend='multiprocessing')(delayed(compute_features_each_seg)(EEG_segs[segi], window_size, channel_num, band_num, NW, Fs, freq, band_freq, total_freq_range, total_freq_id, sub_window_size, sub_step_size, dpss=dpss, eigvals=eigvals) for segi in range(seg_num))
    
    np.set_printoptions(threshold=old_threshold)

    if return_feature_names:
        feature_names = ['mean_gradient_%s'%chn for chn in channel_names]
        feature_names += ['kurtosis_%s'%chn for chn in channel_names]
        feature_names += ['sample_entropy_%s'%chn for chn in channel_names]
        for ffn in ['max','min','mean','std','kurtosis']:#,'skewness'
            for bn in band_names:
                if ffn=='kurtosis' or bn!='sigma': # no need for sigma band
                    feature_names += ['%s_bandpower_%s_%s'%(bn,ffn,chn) for chn in combined_channel_names]

        power_ratios = ['delta/theta','delta/alpha','theta/alpha']
        for pr in power_ratios:
            feature_names += ['%s_max_%s'%(pr,chn) for chn in combined_channel_names]
            feature_names += ['%s_min_%s'%(pr,chn) for chn in combined_channel_names]
            feature_names += ['%s_mean_%s'%(pr,chn) for chn in combined_channel_names]
            feature_names += ['%s_std_%s'%(pr,chn) for chn in combined_channel_names]

    # features.shape = (#epoch, 102)
    
    if return_feature_names:
        return np.array(features), feature_names#, pxx_mts, freqs
    else:
        return np.array(features)#, pxx_mts, freqs

