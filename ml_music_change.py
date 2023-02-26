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
BOSSA_DIR = os.path.join(AUDIO_DIR, 'bossa')
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
                print(file)
                rate, music = read(os.path.join(root, file))
                df_music = pd.DataFrame(music[start:end, :])
                consolidate_music.append(df_music)

    return pd.concat(consolidate_music), rate


if __name__ == '__main__':
    # LSTM configuration
    shape = 100
    iters = 50
    batch = 50

    # set up music range
    data_start = 0
    data_end = 40000

    # creating a new dataframe to train the model
    train_start = data_start
    train_end = 20000

    # creating dataframe from music v1.wav
    # dataset with all musics to train
    consolidate_train, rate = concatenate_music_df(audio_path=BOSSA_DIR,
                                                   consolidate_music=[],
                                                   start=data_start,
                                                   end=data_end)
    # train
    X1, X2, y1, y2 = create_train_dataset(consolidate_train, look_back=3,
                                          train=True)

    # test
    consolidate_test, rate_test = concatenate_music_df(audio_path=BOSSA_DIR,
                                                       consolidate_music=[],
                                                       start=train_start,
                                                       end=train_end)
    test1, test2 = create_train_dataset(
        consolidate_test, look_back=3, train=False)

    # model settings and predictions
    # reshape data
    X1 = X1.reshape((-1, 1, 3))
    X2 = X2.reshape((-1, 1, 3))
    test1 = test1.reshape((-1, 1, 3))
    test2 = test2.reshape((-1, 1, 3))

    # LSTM Model for channel 1 of the music data
    rnn1 = Sequential()
    rnn1.add(LSTM(units=100, activation='relu', input_shape=(None, 3)))
    rnn1.add(Dense(units=50, activation='relu'))
    rnn1.add(Dense(units=25, activation='relu'))
    rnn1.add(Dense(units=12, activation='relu'))
    rnn1.add(Dense(units=1, activation='relu'))
    rnn1.compile(optimizer='adam', loss='mean_squared_error')
    rnn1.fit(X1, y1, epochs=20, batch_size=100)

    # save model

    # making predictions for channel 1 and channel 2
    pred_rnn1 = rnn1.predict(test1)
    pred_rnn2 = rnn1.predict(test2)

    # saving the LSTM predicitons in wav format
    write(os.path.join(RESULT_DIR, 'pred_bossa_rnn.wav'), rate,
          pd.concat([pd.DataFrame(pred_rnn1.astype('int16')),
                     pd.DataFrame(pred_rnn2.astype('int16'))],
                    axis=1).values
          )

    # saving the original music in wav format
    write(os.path.join(AUDIO_DIR, 'bossa_original_lp.wav'), rate,
          consolidate_test.values)

    # LeakyReLu
    rnn1 = Sequential()
    rnn1.add(LSTM(units=100, activation='linear', input_shape=(None, 3)))
    rnn1.add(LeakyReLU())
    rnn1.add(Dense(units=50, activation='linear'))
    rnn1.add(LeakyReLU())
    rnn1.add(Dense(units=25, activation='linear'))
    rnn1.add(LeakyReLU())
    rnn1.add(Dense(units=12, activation='linear'))
    rnn1.add(LeakyReLU())
    rnn1.add(Dense(units=1, activation='linear'))
    rnn1.add(LeakyReLU())
    rnn1.compile(optimizer='adam', loss='mean_squared_error')
    rnn1.fit(X1, y1, epochs=20, batch_size=100)

    # making predictions for channel 1 and channel 2
    pred_rnn1 = rnn1.predict(test1)
    pred_rnn2 = rnn1.predict(test2)

    # save file linear
    write(os.path.join(RESULT_DIR, 'pred_rnn_leaky_relu_linear.wav'), rate,
          pd.concat([pd.DataFrame(pred_rnn1.astype('int16')),
                     pd.DataFrame(pred_rnn2.astype('int16'))],
                    axis=1).values
          )
