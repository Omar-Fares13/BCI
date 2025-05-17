import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Load data for one subject
def load_subject_data(subject_id=1):
    filename = f"patients/BCICIV_2a_{subject_id}.csv"
    data = pd.read_csv(filename)
    return data

# Apply bandpass filter to EEG data
def bandpass_filter(data, lowcut=8, highcut=30, fs=250):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

# Preprocess data
def preprocess_data(data):
    # Extract EEG channels
    eeg_columns = [col for col in data.columns if 'EEG' in col]
    
    # Group by epochs
    epochs = []
    labels = []
    unique_epochs = data['epoch'].unique()
    
    for epoch_id in unique_epochs:
        epoch_data = data[data['epoch'] == epoch_id]
        
        # Skip epochs with missing labels or inconsistent labels
        if epoch_data['label'].nunique() != 1:
            continue
        
        label = epoch_data['label'].iloc[0]
        
        # Extract EEG signals for this epoch
        eeg_signals = epoch_data[eeg_columns].values
        
        # Apply bandpass filtering to each channel
        filtered_signals = np.zeros_like(eeg_signals)
        for i in range(eeg_signals.shape[1]):
            filtered_signals[:, i] = bandpass_filter(eeg_signals[:, i])
        
        epochs.append(filtered_signals)
        labels.append(label)
    
    return np.array(epochs), np.array(labels)