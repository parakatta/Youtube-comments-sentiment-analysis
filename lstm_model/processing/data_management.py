import config.config
import pandas as pd
import tensorflow as tf
def load_dataset(filename):
    dataset = pd.read_csv(filename)
    dataset.head()

    sentences = dataset[config.TEXT].tolist()
    labels = dataset[config.SENTIMENT].tolist()
    return sentences, labels

def load_file(filename):
    return filename

def load_model():
    model = tf.keras.load_model(config.SAVE_PATH+'sentiment.h5')
    return model