import tensorflow as tf

batchsz = 128
total_words = 10000
max_review_len = 200
embedding_len = 100
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)

print(y_test)
print(x_train.shape, len(x_train[0]), y_train.shape)
print(x_test.shape, len(x_test[0]), y_test.shape)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)

word_index = tf.keras.datasets.imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(x_train[0]))
print(y_train[0])


class MyRNN(tf.keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()

        # [b, 64]
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]

        # 词向量编码 [b, 80] => [b, 80, 100]
        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)

        # 构建两个cell
        self.rnn_cell0 = tf.keras.layers.GRUCell(units, dropout=0.5)
        self.rnn_cell1 = tf.keras.layers.GRUCell(units, dropout=0.5)
        # [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        # [b, 80] 输入是多少句话， 每句话80个词
        x = inputs
        # [b, 80] => [b, 80, 100]
        x = self.embedding(x)

        state0 = self.state0
        state1 = self.state1

        # word: [b, 100]
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell0(word, state1, training)

        x = self.outlayer(out1)
        prob = tf.sigmoid(x)

        return prob


def main():
    units = 64
    epochs = 4

    model = MyRNN(units)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(db_train, epochs=epochs, validation_data=db_test)

    model.evaluate(db_test)


if __name__ == '__main__':
    main()
