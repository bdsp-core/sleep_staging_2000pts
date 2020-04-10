#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import Counter
import datetime
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import io as sio
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import MultinomialHMM
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from load_dataset import *
from segment_EEG import *
from extract_features_parallel import *
sys.path.insert(0,'machinelearning_model/pickle_dependencies')
from myelm import *


epoch_time = 30 # [s]
sub_window_time = 2  # [s] for calculating features
sub_window_step = 1  # [s]
start_end_remove_epoch_num = 1
line_freq = 60.  # [Hz]
bandpass_freq = [0.5, 20.]  # [Hz]
amplitude_thres = 500 # [uV]
newFs = 200  # [Hz]
changepoint_epoch_num = 1
sleep_stage_num = 5
combined_EEG_channels = ['F','C','O']
EEG_channels = ['F3-M2','F4-M1','C3-M2','C4-M1','O1-M2','O2-M1']
random_state = 1
normal_only = True


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


def myprint(seg_mask):
    sm = Counter(seg_mask)
    for ex in seg_mask_explanation:
        if ex in sm:
            print('%s: %d/%d, %g%%'%(ex,sm[ex],len(seg_mask),sm[ex]*100./len(seg_mask)))


if __name__=='__main__':
    np.random.seed(random_state)
    ##################
    # use data_list_path to specify which dataset to use
    # data_list.txt:
    # data_path    feature_path    state
    # Signal1.mat  Features1.mat   good
    # ...
    ##################
    data_list_path = 'data/data_list.txt'
    subject_files = pd.read_csv(data_list_path, sep='\t')
    subject_files = subject_files[subject_files.status=='good'].reset_index(drop=True)
    subject_num = len(subject_files)
    
    # load trained model and HMM smoother
    with open('machinelearning_model/elm20000.pickle', 'rb') as f:
        elm_model = pickle.load(f, encoding='latin1')
    with open('machinelearning_model/hmm_smoother.pickle','rb') as f:
        hmm = pickle.load(f, encoding='latin1')
        
    err_subject_reason = []
    err_subject = []
    for si in range(subject_num):
        data_path = subject_files.signal_file.iloc[si]
        feature_path = subject_files.feature_file.iloc[si]
        subject_file_name = os.path.basename(feature_path)
            
        print('\n====== [%d/%d] %s %s ======'%(si+1,subject_num,subject_file_name.replace('.mat',''),datetime.datetime.now()))
        try:
            # check and load dataset
            EEG, params = load_dataset_edf(data_path)
            #EEG, params = load_dataset_mat(data_path)
            Fs = params.get('Fs')

            # segment EEG
            NW = 10
            segs, seg_start_ids, seg_mask, specs, freqs = segment_EEG(EEG, epoch_time, Fs, newFs, NW, amplitude_thres=amplitude_thres, notch_freq=line_freq, bandpass_freq=bandpass_freq, start_end_remove_window_num=start_end_remove_epoch_num, to_remove_mean=False, n_jobs=-1)
            Fs = newFs
			
            if normal_only:
                good_ids = np.where(np.in1d(seg_mask,seg_mask_explanation[:2]))[0]
                if len(good_ids)<=0:
                    myprint(seg_mask)
                    raise ValueError('No normal epochs')
                segs = segs[good_ids]
                seg_start_ids = seg_start_ids[good_ids]
                specs = specs[good_ids]
            else:
                good_ids = np.arange(len(seg_mask))

            # extract features
            NW = 2
            features, feature_names = extract_features(segs, EEG_channels, combined_channel_names, Fs, NW, bandpass_freq, sub_window_time, sub_window_step, return_feature_names=True, n_jobs=-1)
            features[np.isinf(features)] = np.nan
            nan_ids = np.where(np.any(np.isnan(features),axis=1))[0]
            for ii in nan_ids:
                seg_mask[good_ids[ii]] = seg_mask_explanation[8]
            if normal_only and len(nan_ids)>0:
                good_ids2 = np.where(np.in1d(np.array(seg_mask)[good_ids],seg_mask_explanation[:2]))[0]
                segs = segs[good_ids2]
                seg_start_ids = seg_start_ids[good_ids2]
                specs = specs[good_ids2]
                
            myprint(seg_mask)
                
        except Exception as e:
            err_msg = str(e)
            err_info = err_msg.split('\n')[0].strip()
            print('\n%s.\nSubject %s is IGNORED.\n'%(err_info,subject_file_name))
            err_subject_reason.append([subject_file_name,err_info])
            err_subject.append(subject_file_name)
            continue
            
        sio.savemat(feature_path, {'EEG_feature_names':feature_names,
                    'EEG_specs':specs,
                    'EEG_freq':freqs,
                    'EEG_features':features,
                    'seg_start_ids':seg_start_ids.astype(float),
                    'seg_remark':seg_mask})

        features = np.sign(features)*np.log1p(np.abs(features))
        
        # normalized for each patient
        features = StandardScaler().fit_transform(features)
        
        # 1: N3, 2: N2, 3: N1, 4: R, 5: W
        predicted_sleep_stage_prob, predicted_sleep_stages = elm_model.predict_proba(features)
        print(Counter(predicted_sleep_stages))
        
        ## smooth

        predicted_sleep_stages_prob = hmm.predict_proba(predicted_sleep_stages.astype(int).reshape(-1,1)-1)
        predicted_sleep_stages = np.argmax(predicted_sleep_stages_prob, axis=1)+1
        print(Counter(predicted_sleep_stages))
        
        ## visualize
    
        xmax = len(predicted_sleep_stages)*epoch_time/3600.
        specs_db = 10*np.log10(specs)
        specs_db1 = specs_db.mean(axis=1)
        vmax, vmin = np.percentile(specs_db1.flatten(), (98,1))
        
        plt.close()
        fig = plt.figure(figsize=(8,6))
        gs = GridSpec(2, 1, height_ratios=[1,2])
        
        ax1 = fig.add_subplot(gs[0,0])
        ax1.step(np.linspace(0, xmax, len(predicted_sleep_stages)), predicted_sleep_stages)
        ax1.set_ylim([0.5,5.5]); ax1.set_yticks([1,2,3,4,5]); ax1.set_yticklabels(['N3','N2','N1','R','W'])
        ax1.set_xlim([0,xmax])
        ax1.yaxis.grid(True)
        
        ax2 = fig.add_subplot(gs[1,0])
        ax2.imshow(specs_db1.T, cmap='jet', origin='lower',aspect='auto',
                vmax=vmax ,vmin=vmin, extent=(0,xmax,freqs[0],freqs[-1]))
        ax2.set_xlabel('Hour')
                 
        plt.tight_layout()
        #plt.show()
        plt.savefig('sleep_stages_machinelearning_%s.png'%subject_file_name.replace('.mat','').replace('Features_',''),
                    bbox_inches='tight', pad_inches=0.01)
        
        res = sio.loadmat(feature_path)
        res['predicted_sleep_stage_prob'] = predicted_sleep_stage_prob
        res['predicted_sleep_stage'] = predicted_sleep_stages
        res['sleep_stage_text'] = ['N3','N2','N1','R','W']
        sio.savemat(feature_path,res)

