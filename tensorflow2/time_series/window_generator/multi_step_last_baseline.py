import tensorflow as tf


class MultiStepLastBaseline(tf.keras.Model):
    def __init__(self, OUT_STEPS):
        self.OUT_STEPS = OUT_STEPS

    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, self.OUT_STEPS, 1])
