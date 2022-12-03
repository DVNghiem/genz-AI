from abc import ABC
from typing import Union
import tensorflow as tf
import numpy as np
from genz_ai.nlp.layers import EncoderSeq2Seq, DecoderSeq2Seq
from genz_ai.nlp.utils import Config


class Seq2Seq(tf.keras.Model, ABC):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = EncoderSeq2Seq(config)
        self.decoder = DecoderSeq2Seq(config)

        self.enc_unit = config.units
        self.dec_unit = config.units
        self.start_id = config.bos_token_id
        self.end_id = config.eos_token_id
        self.maxlen = config.maxlen

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs):
        super().compile()
        self.loss = loss
        self.optimizer = optimizer
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name='acc', dtype=tf.float32
        )
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name='val_acc', dtype=tf.float32
        )

    @property
    def metrics(self):
        return [self.acc, self.val_acc]

    @tf.function
    def train_step(self, data):
        x, y = data
        loss = 0
        hidden = self.initialize_hidden_state(self.enc_unit, self.batch_size)
        with tf.GradientTape() as tape:
            enc_output, hidden = self.encoder(x, hidden)
            dec_input = tf.expand_dims([self.start_id] * self.batch_size, 1)
            for t in range(1, y.shape[1]):
                predictions, hidden = self.decoder(
                    dec_input, hidden, enc_output)
                loss += self.loss(y[:, t], predictions)
                dec_input = tf.expand_dims(y[:, t], 1)
                self.acc.update_state(y[:, t], predictions)
        batch_loss = (loss / int(y.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return {'loss': batch_loss, 'accuracy': self.acc.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        loss = 0
        hidden = self.initialize_hidden_state(self.enc_unit, self.batch_size)
        enc_output, hidden = self.encoder(x, hidden)
        dec_input = tf.expand_dims([self.start_id] * self.batch_size, 1)
        for t in range(1, y.shape[1]):
            predictions, hidden = self.decoder(
                dec_input, hidden, enc_output)
            loss += self.loss(y[:, t], predictions)
            dec_input = tf.expand_dims(y[:, t], 1)
            self.val_acc.update_state(y[:, t], predictions)
        return {'loss': loss, 'accuracy': self.val_acc.result()}

    def predict(self, x: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
        bs = x.shape[0]
        result = []
        hidden = self.initialize_hidden_state(self.enc_unit, bs)
        enc_out, hidden = self.encoder(x, hidden)
        dec_input = tf.expand_dims([self.start_id] * bs, -1)
        for i in range(self.maxlen):
            predictions, hidden = self.decoder(dec_input,
                                               hidden,
                                               enc_out)
            predicted_id = tf.argmax(predictions, axis=-1).numpy()
            if predicted_id.numpy()[0, 0] == self.config.eos_token_id:
                break
            result.append(predicted_id.numpy()[0, 0])
            dec_input = tf.expand_dims(predicted_id, -1)
        return result

    def initialize_hidden_state(self, hidden, batch_size):
        return tf.zeros(shape=(batch_size, hidden))

    def __str__(self) -> str:
        return 'seq2seq'

    def loadCheckpoint(self, dir: str = 'checkpoint') -> None:
        checkpoint = tf.train.Checkpoint(
            model=self)
        ckpt_manager = tf.train.CheckpointManager(
            checkpoint, dir, max_to_keep=1)
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint)
            print('\nLatest checkpoint restored!!!\n')
