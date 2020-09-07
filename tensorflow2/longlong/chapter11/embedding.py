import tensorflow as tf


x = tf.range(10)
x = tf.random.shuffle(x)

print(x, x.shape)

