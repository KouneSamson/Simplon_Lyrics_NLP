import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

def init_LSTM_encoders() :
    
    lyrics_df = pd.read_csv("src/data_management/lyrics-data.csv",sep=",",quotechar='"')
    artists_df = pd.read_csv("src/data_management/artists-data.csv",sep=",")
    artists_df.rename(columns = {'Link':'ALink'}, inplace = True)
    df = pd.merge(lyrics_df, artists_df, on='ALink', how='right')
    df = df.dropna()
    dfCountry = df.loc[df['Genres'] == 'Country']
    # dfCountry = dfCountry['Lyric'].iloc[0:2500]
    dfCountry = dfCountry['Lyric'].iloc[0:500]
    lyric = ""
    for lyric_data in dfCountry:
        lyric += lyric_data + "\n"
    lyric =lyric.lower()
    chars = sorted(list(set(lyric)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    
    n_chars = len(lyric)
    n_vocab = len(chars)
    
    seq_length = 100
    dataX = []
    dataY = []

    for i in range(0, n_chars - seq_length, 1):
        seq_in = lyric[i:i + seq_length]
        seq_out = lyric[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    X = X / float(n_vocab)
    y = to_categorical(dataY)
    
    return (char_to_int, int_to_char, X, y, n_vocab)
