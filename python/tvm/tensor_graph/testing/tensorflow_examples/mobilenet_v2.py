import tensorflow as tf

import numpy as np

import time
import argparse

def test_train_perf(batch_size):
    model = tf.keras.applications.MobileNetV2(
        input_shape=None, alpha=1.0, include_top=True, weights=None,
        input_tensor=None, pooling='avg', classes=1000, classifier_activation=None
    )

    data = tf.convert_to_tensor(np.random.randn(batch_size, 224, 224, 3), np.float32)
    labels = tf.convert_to_tensor(np.random.randn(batch_size, 1000), np.float32)
    model(data)

    start_time = time.time()
    number = 10
    repeats = 10

    optimizer = tf.optimizers.SGD(learning_rate=0.002)
    for i in range(number):
        records = []
        for j in range(repeats):
            start_time = time.time()
            with tf.GradientTape() as tape:
                logits = model(data)
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            end_time = time.time()
            records.append(end_time - start_time)
        print("Average training latency {} ms".format(1000. * np.mean(records)))
        print("Median training latency {} ms".format(1000. * np.median(records)))

def test_infer_perf(batch_size):
    model = tf.keras.applications.MobileNetV2(
        input_shape=None, alpha=1.0, include_top=True, weights=None,
        input_tensor=None, pooling='avg', classes=1000, classifier_activation=None
    )

    data = tf.convert_to_tensor(np.random.randn(batch_size, 224, 224, 3), np.float32)
    
    model(data)
    
    start_time = time.time()
    number = 10
    repeats = 10
    for i in range(repeats):
        records = []
        for j in range(number):
            start_time = time.time()
            logits = model(data)
            end_time = time.time()
            records.append(end_time - start_time)
        print("Average inference latency {} ms".format(1000. * np.mean(records)))
        print("Median inference latency {} ms".format(1000. * np.median(records)))


if __name__ == "__main__":
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
