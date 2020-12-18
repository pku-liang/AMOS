from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import tf_utils
import tensorflow as tf
import numpy as np

import time
import argparse

class scRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, context_units, n_class, alpha=0.5):
        super(scRNNCell, self).__init__()
        self.units = units
        self.context_units = context_units
        self.alpha = alpha
        self.fc = tf.keras.layers.Dense(units)
        self.classifier = tf.keras.layers.Dense(n_class)

        self.V = self.add_weight(
            shape=(self.context_units, self.units),
            name='V')
        self.U = self.add_weight(
            shape=(self.units, self.units),
            name='U')

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.B = self.add_weight(
            shape=(input_dim, self.context_units),
            name='B')

    def call(self, inputs, states, **kwargs):
        h_tm1, c_tm1 = states

        c = (1 - self.alpha) * (inputs @ self.B) + self.alpha * c_tm1

        h = tf.nn.sigmoid(self.fc(tf.concat([c, inputs, h_tm1], axis=1)))
        h = h @ self.U + c @ self.V

        out = self.classifier(h)

        return out, [h, c]


def test_train_perf(batch_size):
    model = scRNNCell(128, 64, 10)

    data = tf.convert_to_tensor(np.random.randn(batch_size, 28 * 28), np.float32)
    old_h = tf.convert_to_tensor(np.random.randn(batch_size, 128), np.float32)
    old_c = tf.convert_to_tensor(np.random.randn(batch_size, 64), np.float32)
    labels = tf.convert_to_tensor(np.random.randn(batch_size, 10), np.float32)

    @tf.function(experimental_compile=USE_XLA)
    def model_loss(data, states, labels):
        logits, _ = model(data, states)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return loss

    model_loss(data, [old_h, old_c], labels)

    number = 10
    repeats = 10

    optimizer = tf.optimizers.SGD(learning_rate=0.002)
    for i in range(number):
        records = []
        for j in range(repeats):
            start_time = time.time()
            with tf.GradientTape() as tape:
                loss = model_loss(data, [old_h, old_c], labels)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            end_time = time.time()
            records.append(end_time - start_time)
        print("Average training latency {} ms".format(1000. * np.mean(records)))
        print("Median training latency {} ms".format(1000. * np.median(records)))

def test_infer_perf(batch_size):
    model = scRNNCell(128, 64, 10)

    data = tf.convert_to_tensor(np.random.randn(batch_size, 28 * 28), np.float32)
    old_h = tf.convert_to_tensor(np.random.randn(batch_size, 128), np.float32)
    old_c = tf.convert_to_tensor(np.random.randn(batch_size, 64), np.float32)
    labels = tf.convert_to_tensor(np.random.randn(batch_size, 10), np.float32)

    @tf.function(experimental_compile=USE_XLA)
    def model_func(data, states):
        out = model(data, states)
        return out
    model_func(data, [old_h, old_c])
    
    number = 10
    repeats = 10
    for i in range(repeats):
        records = []
        for j in range(number):
            start_time = time.time()
            logits, _ = model_func(data, [old_h, old_c])
            end_time = time.time()
            records.append(end_time - start_time)
        print("Average inference latency {} ms".format(1000. * np.mean(records)))
        print("Median inference latency {} ms".format(1000. * np.median(records)))

if __name__ == "__main__":
    USE_XLA = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=0)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    with tf.device('GPU:'+str(args.device)):
        if args.train:
            test_train_perf(args.batch_size)
        else:
            test_infer_perf(args.batch_size)
