
import numpy as np
import config.config
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Separate out the sentences and labels into training and test sets
def train_test(sentences, labels):
  size = int(len(sentences) * 0.8)
  training_sentences = sentences[0:size]
  testing_sentences = sentences[size:]
  training_labels = labels[0:size]
  testing_labels = labels[size:]
  train_labels = np.array(training_labels)
  test_labels = np.array(testing_labels)
  return train_labels, test_labels, training_sentences, testing_sentences


def tokenize_seq_padding(training_sentences, testing_sentences, tokenizer):
    
  tokenizer.fit_on_texts(training_sentences)
  word_index = tokenizer.word_index
  train_sequences = tokenizer.texts_to_sequences(training_sentences)
  train_padded = pad_sequences(train_sequences,maxlen=config.max_length,padding= 'post',truncating='post')
  test_sequences= tokenizer.texts_to_sequences(testing_sentences)
  test_padded = pad_sequences(test_sequences,maxlen= config.max_length, padding='post',truncating='post')
  return train_padded, test_padded

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
    
def save_model(model):
      model.save(config.SAVE_PATH+'sentiment.h5')
      

