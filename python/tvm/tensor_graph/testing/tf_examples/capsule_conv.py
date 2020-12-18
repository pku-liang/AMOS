import tensorflow as tf
import numpy as np
import time
from tensorflow.keras import datasets, layers, models


def get_model(inputs, M, N, K):
  out_lst = []
  convs = []
  for k in range(K):
    tmp = []
    for j in range(N):
      tmp.append(tf.keras.layers.Conv2D(32, 9, padding="valid", strides=2))
    convs.append(tmp)
  
  for i in range(M):
    for j in range(N):
      out_ij = 0.0
      for k in range(K):
        x_ik = tf.slice(inputs, [i, k, 0, 0, 0, 0], [1, 1, *inputs.shape[2:]])
        x_ik = tf.reshape(x_ik, inputs.shape[2:])
        out_ij = out_ij + convs[k][j](x_ik)
      out_lst.append(out_ij)
  out = tf.stack(out_lst)
  out = tf.reshape(out, [M, N, *out.shape[1:]])
  print(out.shape)
  return out


if __name__ == "__main__":
  with tf.device('GPU:0'):
    tf.compat.v1.disable_eager_execution()
    data_shape = [1, 1, 1, 28, 28, 256]
    inputs = tf.compat.v1.placeholder(tf.float32, data_shape)
    output = get_model(inputs, 1, 8, 1)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    inputs_np = np.random.uniform(-1, 1, data_shape)
    outputs = sess.run(output, feed_dict={inputs: inputs_np})
    trials = 1000
    beg = time.time()
    for i in range(trials):
      outputs = sess.run(output, feed_dict={inputs: inputs_np})
    end = time.time()
    print("end-to-end time cost=", (end - beg) * 1e3 / trials, "ms")
    sess.close()