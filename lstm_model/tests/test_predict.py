
import math

from predict import predict_review
from processing.data_management import load_file, load_model
import tensorflow_datasets as tfds
import config.config

def test_make_single_prediction():
    # Given
    test_data = load_file(filename = config.TEST_FILE)
    tokenizer1 = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(test_data, config.vocab_size, max_subword_length=5)
    tokenizer1.save_to_file('tokenized_vocab')
    model = load_model()

    # When
    subject = predict_review(model, test_data, tokenizer1)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
    assert math.ceil(subject.get('predictions')[0]) == 112476