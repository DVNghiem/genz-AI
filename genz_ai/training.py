import tensorflow as tf
from .utils import CallbackSave
from typing import List, Any


class TrainArgument:
    def __init__(
            self,
            optimizer: tf.keras.optimizers.Optimizer,
            loss: Any,
            model_dir='model',
            epochs=10,
            batch_size=32,
            save_per_epochs=1):
        self.model_dir = model_dir
        self.epochs = epochs
        self.bach_size = batch_size
        self.save_per_epochs = save_per_epochs
        self.optimizer = optimizer
        self.loss = loss


class Training:
    def __init__(
            self,
            model: tf.keras.Model,
            train_arg: TrainArgument,
            data_train: tf.data.Dataset = None,
            data_eval: tf.data.Dataset = None,
            callbacks: List[tf.keras.callbacks.Callback] = [],
    ):
        self.model = model
        self.data_train = data_train
        self.data_eval = data_eval
        self.optimizer = train_arg.optimizer
        self.train_arg = train_arg
        self.callbacks = callbacks

        self.model.batch_size = train_arg.bach_size
        self.model.compile(loss=train_arg.loss, optimizer=self.optimizer)
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, train_arg.model_dir, max_to_keep=1)

    def train(self):
        if self.ckpt_manager.latest_checkpoint:
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            print("Checkpoint restored")
        self.model.fit(
            self.data_train,
            epochs=self.train_arg.epochs,
            validation_data=self.data_eval,
            callbacks=[CallbackSave(self.ckpt_manager, self.train_arg.save_per_epochs), *self.callbacks]
        )

    def save(self):
        self.ckpt_manager.save()
