import tensorflow as tf

a = tf.range(10, dtype=tf.int32)
print(a)
b = a[:, tf.newaxis]
print(b)

c = a[tf.newaxis, :]
print(c)
