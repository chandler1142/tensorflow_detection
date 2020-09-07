import tensorflow as tf
import tensorflow.keras.layers as layers

cell = layers.SimpleRNNCell(3)
cell.build(input_shape=(None, 4))
cell.trainable_variables

h0 = tf.zeros([4, 64])
x = tf.random.normal([4, 80, 100])
xt = x[:, 0, :]
print(xt.shape)
cell = layers.SimpleRNNCell(64)

