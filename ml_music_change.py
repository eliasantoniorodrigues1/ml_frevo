from scipy.io.wavfile import read, write
from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU, Conv1D, Flatten
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'


# base dir to get audio data projetc
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, 'audio')
FREVO_DIR = os.path.join(AUDIO_DIR, 'frevo')
RESULT_DIR = os.path.join(BASE_DIR, 'result')


# function to create training data by shifting the music data
def create_train_dataset(df, look_back, train=True):
    dataX1, dataX2, dataY1, dataY2 = [], [], [], []
    for i in range(len(df)-look_back-1):
        dataX1.append(df.iloc[i: i + look_back, 0].values)
        dataX2.append(df.iloc[i: i + look_back, 1].values)
        if train:
            dataY1.append(df.iloc[i + look_back, 0])
            dataY2.append(df.iloc[i + look_back, 1])

    if train:
        # return dataset train
        return np.array(dataX1), np.array(dataX2), np.array(dataY1), np.array(dataY2)
    else:
        # return dataset test
        return np.array(dataX1), np.array(dataX2)


def concatenate_music_df(audio_path: str, consolidate_music: list,
                         start: int,
                         end: int) -> pd.DataFrame:
    '''
        this function to consolidate all music in the folder audio
        and returns a list of datasets.
        you need to send a folder containing wav files
        params: full_path_file: string containing the audio path
        params: consolidate_music: empty list to be filled
        params: rate: data generate by music read
        returns consolidate_music, rate
    '''

    # search audions in the path
    for root, _, files in os.walk(audio_path):
        for file in files:
            if file.endswith('.wav'):
                rate, music = read(os.path.join(root, file))
                df_music = pd.DataFrame(music[start:end, :])
                consolidate_music.append(df_music)

    return pd.concat(consolidate_music), rate
