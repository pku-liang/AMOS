from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import tf_utils
import tensorflow as tf
import numpy as np

import time
import argparse

def test_train_perf(model, data_shape, h_shape, c_shape, labels_shape, config):
    assert (h_shape == None and c_shape == None) or (h_shape != None and c_shape != None)
    data_val = np.random.randn(*data_shape)
    labels_val = np.random.randn(*labels_shape)

    data = tf.placeholder(tf.float32, shape=data_shape)
    labels = tf.placeholder(tf.float32, shape=labels_shape)

    values = {
        data: data_val,
        labels: labels_val
    }
    
    if h_shape != None:
        old_h_val = np.random.randn(*h_shape)
        old_c_val = np.random.randn(*c_shape)
        old_h = tf.placeholder(tf.float32, shape=h_shape)
        old_c = tf.placeholder(tf.float32, shape=c_shape)
        values.update({old_h: old_h_val})
        values.update({old_c: old_c_val})
        inputs = data, [old_h, old_c]
    else:
        inputs = data,
    
    optimizer = tf.train.GradientDescentOptimizer(0.002)

    logits, _ = model(*inputs)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    updates = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    loss_val, _ = sess.run([loss, updates], feed_dict=values)

    number = 10
    repeats = 10

    for i in range(number):
        records = []
        for j in range(repeats):
            start_time = time.time()
            loss_val, _ = sess.run([loss, updates], feed_dict=values)
            end_time = time.time()
            records.append(end_time - start_time)
        print("Average training latency {} ms".format(1000. * np.mean(records)))
        print("Median training latency {} ms".format(1000. * np.median(records)))

def test_infer_perf(model, data_shape, h_shape, c_shape, config):
    assert (h_shape == None and c_shape == None) or (h_shape != None and c_shape != None)
    data_val = np.random.randn(*data_shape)

    data = tf.placeholder(tf.float32, shape=data_shape)

    values = {
        data: data_val
    }
    
    if h_shape != None:
        old_h_val = np.random.randn(*h_shape)
        old_c_val = np.random.randn(*c_shape)
        old_h = tf.placeholder(tf.float32, shape=h_shape)
        old_c = tf.placeholder(tf.float32, shape=c_shape)
        values.update({old_h: old_h_val})
        values.update({old_c: old_c_val})
        inputs = data, [old_h, old_c]
    else:
        inputs = data,
    
    out, _ = model(*inputs)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    out_val = sess.run([out], feed_dict=values)
    
    number = 10
    repeats = 10
    for i in range(repeats):
        records = []
        for j in range(number):
            start_time = time.time()
            out_val = sess.run([out], feed_dict=values)
            end_time = time.time()
            records.append(end_time - start_time)
        print("Average inference latency {} ms".format(1000. * np.mean(records)))
        print("Median inference latency {} ms".format(1000. * np.median(records)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", type=str, default="subLSTM")
    parser.add_argument("--xla", action="store_true")
    parser.add_argument("--nGraph", action="store_true")
    args = parser.parse_args()

    model = None
    data_shape = None
    h_shape = None
    c_shape = None
    labels_shape = None
    if args.model == "subLSTM":
        from subLSTM import subLSTMCell
        model = subLSTMCell(128, 10)
        data_shape = (args.batch_size, 28*28)
        h_shape = (args.batch_size, 128)
        c_shape = (args.batch_size, 128)
        labels_shape = (args.batch_size, 10)
    elif args.model == "scRNN":
        from scRNN import scRNNCell
        model = scRNNCell(128, 64, 10)
        data_shape = (args.batch_size, 28*28)
        h_shape = (args.batch_size, 128)
        c_shape = (args.batch_size, 64)
        labels_shape = (args.batch_size, 10)
    elif args.model == "LLTM":
        from LLTM import LLTMCell
        model = LLTMCell(128, 10)
        data_shape = (args.batch_size, 28*28)
        h_shape = (args.batch_size, 128)
        c_shape = (args.batch_size, 128)
        labels_shape = (args.batch_size, 10)
    elif args.model == "MILSTM":
        from MILSTM import MILSTMCell
        model = MILSTMCell(1024, 10)
        data_shape = (args.batch_size, 28*28)
        h_shape = (args.batch_size, 1024)
        c_shape = (args.batch_size, 1024)
        labels_shape = (args.batch_size, 10)
    elif args.model == "ShuffleNet":
        from shufflenet_v1 import ShuffleNet
        model = ShuffleNet
        data_shape = (args.batch_size, 224, 224, 3)
        labels_shape = (args.batch_size, 1000)
    
    config = None
    if args.xla == True:
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    if args.nGraph == True:
        assert config == None
        import ngraph_bridge
        ngraph_bridge.list_backends()
        ngraph_bridge.set_backend('PLAIDML')
        config = tf.ConfigProto()
        config = ngraph_bridge.update_config(config)

    with tf.device('GPU:'+str(args.device)):
        if args.train:
            test_train_perf(model, data_shape, h_shape, c_shape, labels_shape, config)
        else:
            test_infer_perf(model, data_shape, h_shape, c_shape, config)
