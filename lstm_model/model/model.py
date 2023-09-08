

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import Tokenizer and pad_sequences
import tensorflow as tf
import config.config
import tensorflow_datasets as tfds
"""## Define LSTM model"""

def make_model():
    model_lstm=tf.keras.Sequential([
        tf.keras.layers.Embedding(config.vocab_size,config.embedding_dim,input_length=config.max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.embedding_dim)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model_lstm.summary()
    return model_lstm