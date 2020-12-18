from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import tf_utils
import tensorflow as tf
import numpy as np

import numpy as np

import time
import argparse

class MILSTMCell(tf.keras.layers.Layer):
  def __init__(self,
               units,
               n_class,
               **kwargs):
    super(MILSTMCell, self).__init__(**kwargs)
    self.units = units

    self.kernel = tf.keras.layers.Dense(self.units * 4)
    self.recurrent_kernel = tf.keras.layers.Dense(self.units * 4)

    self.alpha = self.add_weight(
        shape=(1, self.units * 4),
        name='alpha')
    self.beta1 = self.add_weight(
        shape=(1, self.units * 4),
        name='beta1')
    self.beta2 = self.add_weight(
        shape=(1, self.units * 4),
        name='beta2')

    self.classifier = tf.keras.layers.Dense(n_class)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1]

  def call(self, inputs, states):
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    x = self.kernel(inputs)
    h_tmp = self.recurrent_kernel(h_tm1)

    xh_alpha = self.alpha * x * h_tmp
    x_beta1 = self.beta1 * x
    h_beta2 = self.beta2 * h_tmp

    i, f, c, o = tf.split(xh_alpha + x_beta1 + h_beta2, num_or_size_splits=4, axis=1)

    i = activations.sigmoid(i)
    f = activations.sigmoid(f)
    o = activations.sigmoid(o)
    c = activations.tanh(c) * i + c_tm1 * f

    h = activations.tanh(c) * o
    out = self.classifier(h)
    return out, [h, c]


def test_train_perf(batch_size):
    model = MILSTMCell(1024, 10)

    data = tf.convert_to_tensor(np.random.randn(batch_size, 28 * 28), np.float32)
    old_h = tf.convert_to_tensor(np.random.randn(batch_size, 1024), np.float32)
    old_c = tf.convert_to_tensor(np.random.randn(batch_size, 1024), np.float32)
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
    model = MILSTMCell(1024, 10)
    data = tf.convert_to_tensor(np.random.randn(batch_size, 28 * 28), np.float32)
    old_h = tf.convert_to_tensor(np.random.randn(batch_size, 1024), np.float32)
    old_c = tf.convert_to_tensor(np.random.randn(batch_size, 1024), np.float32)
    
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
