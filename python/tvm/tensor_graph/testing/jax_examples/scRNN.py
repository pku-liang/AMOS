# %%
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad, random, lax
from jax.nn import sigmoid, log_softmax
from jax.nn.initializers import glorot_normal, normal
from jax.experimental import stax, optimizers
from jax.experimental.stax import BatchNorm, Conv, Dense, Flatten, \
                                   Relu, LogSoftmax
from jax.scipy.special import logsumexp

import numpy as onp
from functools import partial
import time
import argparse

def scRNN(units, context_units, num_classes, alpha=0.5, W_init=glorot_normal(), b_init=normal()):
    def init_fun(rng, input_shape):
        k1, k2 = random.split(rng)
        hidden, cell = (b_init(k1, (input_shape[0], units)),
                        b_init(k2, (input_shape[0], context_units)))

        k1, k2, k3 = random.split(k1, num=3)
        B, U, V = (
            W_init(k1, (input_shape[1], context_units)),
            W_init(k2, (units, units)),
            W_init(k3, (context_units, units)),
        )

        k1, k2 = random.split(k1, num=2)
        W, b = (
            W_init(k1, (context_units + input_shape[1] + units, units)),
            b_init(k2, (units,)),
        )

        k1, k2 = random.split(k1, num=2)
        class_W, class_b = (
            W_init(k1, (units, num_classes)),
            b_init(k2, (num_classes,)),
        )

        output_shape = (input_shape[0], num_classes)
        return (output_shape,
                ((hidden, cell),
                 (B, U, V),
                 (W, b),
                 (class_W, class_b),
                 alpha),)

    def apply_fun(params, inp, **kwargs):
        (hidden, cell), (B, U, V), (W, b), (class_W, class_b), alpha = params

        cell = (1 - alpha) * np.dot(inp, B) + alpha * cell
        hidden = sigmoid(np.dot(np.concatenate([cell, inp, hidden], axis=1), W) + b)
        hidden = np.dot(hidden, U) + np.dot(cell, V)
        out = np.dot(hidden, class_W) + class_b
        return out

    return init_fun, apply_fun

def cross_entropy_loss(params, inputs, targets, model):
    logits = model(params, inputs)
    return -np.sum(log_softmax(logits) * targets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    # Generate key which is used to generate random numbers
    key = random.PRNGKey(1)

    batch_size = args.batch_size
    in_dim = 28*28
    num_hidden_units = 128
    num_context_units = 64
    num_classes = 10

    key, data_key, labels_key = random.split(key, 3)
    data_shape = (batch_size, in_dim)
    labels_shape = (batch_size, num_classes)
    data = random.normal(data_key, data_shape)
    labels = random.normal(labels_key, labels_shape)

    # Initialize the network and perform a forward pass
    init_fun, apply_scRNN = scRNN(num_hidden_units, num_context_units, num_classes)
    _, params = init_fun(key, data_shape)

    step_size = 0.002
    opt_init, opt_update, get_params = optimizers.sgd(step_size)
    opt_state = opt_init(params)

    @jit
    def update(i, opt_state, inputs, targets):
        params = get_params(opt_state)
        return opt_update(i, grad(cross_entropy_loss)(params, inputs, targets, apply_scRNN), opt_state)

    opt_state = update(0, opt_state, data, labels)

    number = 10
    repeats = 10

    for i in range(number):
        records = []
        for j in range(repeats):
            start_time = time.time()
            opt_state = update(j, opt_state, data, labels)
            end_time = time.time()
            records.append(end_time - start_time)
        print("Average training latency {} ms".format(1000. * onp.mean(records)))
        print("Median training latency {} ms".format(1000. * onp.median(records)))

    print('-' * 48)

    logits = jit(apply_scRNN)(params, data)

    for i in range(number):
        records = []
        for j in range(repeats):
            start_time = time.time()
            logits = jit(apply_scRNN)(params, data)
            end_time = time.time()
            records.append(end_time - start_time)
        print("Average inference latency {} ms".format(1000. * onp.mean(records)))
        print("Median inference latency {} ms".format(1000. * onp.median(records)))

    print(jax.local_devices())