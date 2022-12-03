from typing import Any, Union
import tensorflow as tf
import numpy as np
from genz_ai.nlp.layers import Encoder, Decoder, PositionEmbedding
from genz_ai.nlp.utils import Config, create_look_ahead_mask, create_padding_mask

class Transformer(tf.keras.Model):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.config = config

        self.embedding_ec = PositionEmbedding(
            config.input_vocab_size, config.input_vocab_size, config.maxlen)

        if config.num_lang == 2:
            self.embedding_de = PositionEmbedding(
                config.target_vocab_size, config.hidden_size, config.maxlen)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.final_layer = tf.keras.layers.Dense(config.target_vocab_size)

    def call(self, x):
        inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask = x
        inp = self.embedding_ec(inp)
        if self.config.num_lang == 1:
            tar = self.embedding_ec(tar)
        else:
            tar = self.embedding_de(tar)
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)

        return final_output

    def create_masks(self, inp, tar) -> Any:
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask

    def compile(self, loss, optimizer) -> None:
        super(Transformer, self).compile()
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
        inp, tar = data
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
            inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = self((inp, tar_inp,
                               True,
                               enc_padding_mask,
                               combined_mask,
                               dec_padding_mask))
            loss = self.loss(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        self.acc.update_state(tar_real, predictions)
        return {'loss': loss, 'accuracy': self.acc.result()}

    @tf.function
    def test_step(self, data):
        inp, tar = data
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
            inp, tar_inp)
        predictions = self((inp, tar_inp,
                            False,
                            enc_padding_mask,
                            combined_mask,
                            dec_padding_mask))
        loss = self.loss(tar_real, predictions)

        self.val_acc.update_state(tar_real, predictions)
        return {'loss': loss, 'accuracy': self.val_acc.result()}

    def predict(self,  x: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
        result = []
        output = tf.expand_dims([self.config.bos_token_id], axis=-1)
        for i in range(self.config.maxlen):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
                x, output)
            predictions = self((x,
                                output,
                                False,
                                enc_padding_mask,
                                combined_mask,
                                dec_padding_mask))
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(
                tf.argmax(predictions, axis=-1), dtype=tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)
            if predicted_id.numpy()[0, 0] == self.config.eos_token_id:
                break
            result.append(predicted_id.numpy()[0, 0])
        return result

    def loadCheckpoint(self, dir: str = 'checkpoint') -> None:
        checkpoint = tf.train.Checkpoint(
            model=self)
        ckpt_manager = tf.train.CheckpointManager(
            checkpoint, dir, max_to_keep=1)
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint)
            print('\nLatest checkpoint restored!!!\n')

    def __str__(self) -> str:
        return 'transformer'


class TransformerClassification(tf.keras.Model):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.encoder = Encoder(config)
        self.global_conv = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.fc = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(
            config.num_class, activation='softmax')

    def call(self, x):
        inp, training, enc_padding_mask = x
        enc_output = self.encoder(inp, training, enc_padding_mask)
        x = self.global_conv(enc_output)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.out(x)
        return x

    def compile(self, loss, optimizer) -> None:
        super().compile()
        self.loss = loss
        self.optimizer = optimizer
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name="acc", dtype=tf.float32
        )
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_acc", dtype=tf.float32
        )

    @property
    def metrics(self):
        return [self.acc, self.val_acc]

    @tf.function
    def train_step(self, data):
        x, y = data
        mask = create_padding_mask(x)
        with tf.GradientTape() as tape:
            predict = self((x, True, mask))
            loss = self.loss(y, predict)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        self.acc.update_state(y, predict)
        return {'loss': loss, 'accuracy': self.acc.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        predict = self.predict(x)
        loss = self.loss(y, predict)
        self.val_acc.update_state(y, predict)
        return {'loss': loss, 'accuracy': self.val_acc.result()}

    def predict(self, x: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
        mask = create_padding_mask(x)
        predict = self((x, False, mask))
        return predict

    def loadCheckpoint(self, dir: str = 'checkpoint'):
        checkpoint = tf.train.Checkpoint(
            model=self)
        ckpt_manager = tf.train.CheckpointManager(
            checkpoint, dir, max_to_keep=1)
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint)
            print('\nLatest checkpoint restored!!!\n')

    def __str__(self) -> str:
        return 'transformer_cls'