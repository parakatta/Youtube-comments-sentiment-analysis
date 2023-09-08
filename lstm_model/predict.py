import config.config
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_review(model, new_sentences, tokenizer1 , maxlen=config.max_length, show_padded_sequence=True):
    new_sequences = []
    for i, frvw in enumerate(new_sentences):
        new_sequences.append(tokenizer1.encode(frvw))
    new_reviews_padded = pad_sequences(new_sequences, maxlen=maxlen,
                                    padding='post', truncating='post')
    classes = model.predict(new_reviews_padded)
    for x in range(len(new_sentences)):
        if (show_padded_sequence):
            print(new_reviews_padded[x])
            print(new_sentences[x])
            print(classes[x])
            print("\n")