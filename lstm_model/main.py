# -*- coding: utf-8 -*-
"""sentiment-analysis-in-nlp-lstm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sgCg1EM0tjbiJ3lcx1Tst0CjeeZtVDVz
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import Tokenizer and pad_sequences
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from preprocessing.data_management import load_dataset, load_file, load_model
import preprocessing.preprocessors as pp
import config.config
import tensorflow_datasets as tfds
import model.model as m
import predict
"""## Get the dataset"""

def run_training():
  sentences, labels = load_dataset(config.DATASET)
  tokenizer1 = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(sentences, config.vocab_size, max_subword_length=5)
  tokenizer1.save_to_file('tokenized_vocab')
  tokenizer = Tokenizer(num_words= config.vocab_size, oov_token='<oov>')

  """## Train test split"""
  train_labels, test_labels, training_sentences, testing_sentences = pp.train_test(sentences, labels)

  """## Tokenizer, Sequences, Padding"""
  train_padded, test_padded = pp.tokenize_seq_padding(training_sentences, testing_sentences, tokenizer)

  model = m.make_model()
  history = model.fit(train_padded, train_labels, epochs=config.epochs,
                        validation_data=(test_padded,test_labels))
  pp.save_model(model)
  pp.plot_graphs(history, "accuracy")
  pp.plot_graphs(history, "loss")
  return model

if __name__=='__main__':
    
    model, tokenizer1 = run_training()
    #Test Prediction
    test_data = load_file(filename=config.TEST_FILE)
    predict.predict_review(model, test_data, tokenizer1)





