import tensorflow as tf
import tensorflow.keras.layers as layers

x = tf.random.normal([2, 80, 100])
layer = layers.LSTM(64)
out = layer(x)

print(out.shape)
