import unittest
from genz_ai.nlp.models import Seq2Seq, Transformer
from genz_ai.nlp.utils import loss_seq2seq, loss_transformer
from genz_ai.training import TrainArgument, Training
from genz_ai.nlp.config import Config
import tensorflow as tf
import numpy as np


class NLP(unittest.TestCase):

    @staticmethod
    def test_seq2seq():
        train_x = np.random.random(size=(3, 10))
        train_y = np.random.random(size=(3, 5))
        dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        batch_size = 2
        dataset = dataset.batch(batch_size, drop_remainder=True)
        arg = TrainArgument(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss_seq2seq,
            batch_size=batch_size,
            epochs=2
        )
        config = Config(input_vocab_size=10, target_vocab_size=10, hidden_size=10, num_heads=2, num_hidden_layers=2)
        model = Seq2Seq(config)
        training = Training(model, arg, data_train=dataset)
        training.train()

    @staticmethod
    def test_transformer():
        train_x = np.random.random(size=(3, 10))
        train_y = np.random.random(size=(3, 5))
        dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        batch_size = 2
        dataset = dataset.batch(batch_size, drop_remainder=True)
        arg = TrainArgument(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss_transformer,
            batch_size=batch_size,
            epochs=2
        )
        config = Config(input_vocab_size=10, target_vocab_size=10, hidden_size=10, num_heads=2, num_hidden_layers=2)
        model = Transformer(config)
        training = Training(model, arg, data_train=dataset)
        training.train()


if __name__ == '__main__':
    unittest.main(verbosity=2)
