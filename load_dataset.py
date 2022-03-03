#TODO THIS FILE HAS TO BE MODIFIED BASED ON YOUR OWN DATA FORMAT
import numpy as np
import scipy.io as sio
import mne


def load_dataset_mat(data_path):

    # load data
    try:
        res = sio.loadmat(data_path)
    except Exception as ee:
        import mat73
        res = mat73.loadmat(data_path)
    eeg = res['s'][:6]
    params = {'Fs':200}
    
    return eeg, params
    
    
def load_dataset_edf(data_path):
    
    edf = mne.io.read_raw_edf(data_path, stim_channel=None, preload=True)
    
    # signal.shape = (#channel, T)
    signal = edf.get_data()*1e6
    
    # make sure the output signal has the following channels:
    # F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1
    #channels = edf.get_info()['ch_names']
    #channels = edf.get_info()['Label']
    #F3_id = channels.index('F3')
    #F4_id = channels.index('F4')
    #C3_id = channels.index('C3')
    #C4_id = channels.index('C4')
    #O1_id = channels.index('O1')
    #O2_id = channels.index('O2')
    #M1_id = channels.index('M1')  # sometimes it is called "A1"
    #M2_id = channels.index('M2')  # sometimes it is called "A2"
    
    F3_id = 0
    F4_id = 1
    C3_id = 2
    C4_id = 3
    O1_id = 4
    O2_id = 5
    M1_id = 6
    M2_id = 7
        
    eeg = np.array([
        signal[F3_id] - signal[M2_id],
        signal[F4_id] - signal[M1_id],
        signal[C3_id] - signal[M2_id],
        signal[C4_id] - signal[M1_id],
        signal[O1_id] - signal[M2_id],
        signal[O2_id] - signal[M1_id],
    ])
    # eeg.shape = (6, T)
    
    # get sampling frequency Fs
    Fs = edf.info['sfreq']
    params = {'Fs':Fs}    
    
    return eeg, params

