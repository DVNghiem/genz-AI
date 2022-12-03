import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def get_config(self):
        super().get_config()

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



class CallbackSave(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_manager: tf.train.CheckpointManager, save_per_epochs: int = 1) -> None:
        super().__init__()
        self.ckpt_manager = ckpt_manager
        self.save_per_epochs = save_per_epochs

    def on_epoch_end(self, epochs, logs=None):
        if epochs % self.save_per_epochs == 0:
            self.ckpt_manager.save()