from ml_music_change import concatenate_music_df, create_train_dataset
from ml_music_change import FREVO_DIR, RESULT_DIR, AUDIO_DIR
from scipy.io.wavfile import read, write
from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU, Conv1D, Flatten
import pandas as pd
import os


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

    #---------------------- CREATING DATASETS X, Y ---------------------------#
    # creating dataframe from music v1.wav
    # dataset with all musics to train
    consolidate_train, rate = concatenate_music_df(audio_path=FREVO_DIR,
                                                   consolidate_music=[],
                                                   start=data_start,
                                                   end=data_end)
    # train
    X1, X2, y1, y2 = create_train_dataset(consolidate_train, look_back=3,
                                          train=True)

    # test
    consolidate_test, rate_test = concatenate_music_df(audio_path=FREVO_DIR,
                                                       consolidate_music=[],
                                                       start=train_start,
                                                       end=train_end)
    test1, test2 = create_train_dataset(
        consolidate_test, look_back=3, train=False)
    #-------------------------------------------------------------------------#

    #---------------------- MODEL PREPARATION AND TRAINING -------------------#
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

    # making predictions for channel 1 and channel 2
    pred_rnn1 = rnn1.predict(test1)
    pred_rnn2 = rnn1.predict(test2)

    # saving the LSTM predicitons in wav format
    write(os.path.join(RESULT_DIR, 'pred_frevo_rnn.wav'), rate,
          pd.concat([pd.DataFrame(pred_rnn1.astype('int16')),
                     pd.DataFrame(pred_rnn2.astype('int16'))],
                    axis=1).values
          )

    # saving the original music in wav format
    write(os.path.join(AUDIO_DIR, 'frevo_original_lp.wav'), rate,
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

    #-------------------------------------------------------------------------#
    # save model
    rnn1.save(filepath='ml_frevo_model_lstm')
    # saver = rnn1.train.Saver()
    # saver.save(sess, 'ml_frevo_model_lstm')

    # restore model
    # saver = rnn1.train.Saver()
    # saver.restore(sess, 'ml_frevo_model_lstm')
    # y_pred = model.predict(X_test)
    