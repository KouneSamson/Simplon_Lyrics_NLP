import numpy as np
from progress.bar import PixelBar
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM

from src.data_management.data_encoder import init_LSTM_encoders

char_to_int, int_to_char, X, y, n_vocab = init_LSTM_encoders()

def load_LSTM_model():
    global X
    global y
    
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    # model.load_weights("src/weight_models/weights-improvement-20-1.4644-bigger.hdf5")
    # model.load_weights("src/weight_models/weights-improvement-25-1.4486-bigger.hdf5")
    model.load_weights("src/weight_models/weights-improvement-42-1.4575-bigger.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

def LSTM_predict(phrase : str, model):
    global n_vocab
    global char_to_int
    global int_to_char
    
    pattern = [char_to_int[char] for char in phrase]
    big_result = pattern
    bar = PixelBar('Processing :: ', max=1000)
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        big_result.append(index)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        bar.next()
    bar.finish()
    
    seq_out = [int_to_char[value] for value in big_result]
    lyric = ""
    for item in seq_out:
        lyric += item
    return dict(seq_in=phrase,result=lyric)

def load_GPT2_model():
    return None

def GPT2_predict(phrase : str, model):
    return None