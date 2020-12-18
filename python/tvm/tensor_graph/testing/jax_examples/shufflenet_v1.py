# %%
import jax
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad, random, lax
from jax.nn import sigmoid, log_softmax
from jax.nn.initializers import glorot_normal, normal
from jax.experimental import stax, optimizers
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten, Identity,
                                   Relu, LogSoftmax, AvgPool, FanInConcat,
                                   FanOut, FanInConcat, FanInSum)
from jax.scipy.special import logsumexp

import numpy as onp
from functools import partial
import time
import argparse

def shuffle_channels(num_groups):
    def init_fun(rng, input_shape, **kwargs):
        return (input_shape, ())
    
    def apply_fun(params, x, **kwargs):
        n, h, w, c = x.shape
        x_reshaped = np.reshape(x, [-1, h, w, num_groups, c // num_groups])
        x_transposed = np.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = np.reshape(x_transposed, [-1, h, w, c])
        return x
    return init_fun, apply_fun

def FanSplit(num_groups=None):
    def init_fun(rng, input_shape, **kwargs):
        n, h, w, c = input_shape
        assert c % num_groups == 0
        shape = (n, h, w, c//num_groups)
        output_shape = [shape] * num_groups
        return (
            output_shape,
            ()
        )
    def apply_fun(params, inputs, **kwargs):
        return np.split(inputs, num_groups, axis=3)
    return init_fun, apply_fun

def GroupConv(filters, kernel, strides, num_groups):
    assert filters % num_groups == 0
    layers = [FanSplit(num_groups), 
        stax.parallel(*[Conv(filters//num_groups, kernel, strides=strides, padding='SAME') for _ in range(num_groups)]),
        FanInConcat(axis=3)]
    return stax.serial(*layers)

def ShuffleNetUnitA(in_channels, out_channels, num_groups=3):
    assert in_channels == out_channels
    assert out_channels % 4 == 0
    bottleneck_channels = out_channels // 4

    Block = stax.serial(GroupConv(bottleneck_channels, (1, 1), (1, 1), num_groups),
        BatchNorm(), Relu,
        shuffle_channels(num_groups),
        GroupConv(bottleneck_channels, (3, 3), (1, 1), bottleneck_channels),
        BatchNorm(),
        GroupConv(out_channels, (1, 1), (1, 1), num_groups),
        BatchNorm()
    )

    return stax.serial(FanOut(2), stax.parallel(Block, Identity), FanInSum, Relu)

def ShuffleNetUnitB(in_channels, out_channels, num_groups=3):
    out_channels -= in_channels
    assert out_channels % 4 == 0
    bottleneck_channels = out_channels // 4
    
    layers = [GroupConv(bottleneck_channels, (1, 1), (1, 1), num_groups),
        BatchNorm(), Relu,
        shuffle_channels(num_groups),
        GroupConv(bottleneck_channels, (3, 3), (2, 2), bottleneck_channels),
        BatchNorm(),
        GroupConv(out_channels, (1, 1), (1, 1), num_groups),
        BatchNorm()
    ]
    Block = stax.serial(*layers)

    return stax.serial(FanOut(2), stax.parallel(Block, AvgPool((3, 3), (2, 2), 'SAME')), FanInConcat(axis=-1), Relu)

def ShuffleNet(groups=3, in_channels=3, num_classes=1000):
    layers = [Conv(24, (3, 3), (2, 2), 'SAME')] + \
        [AvgPool((3, 3), strides=(2, 2), padding='SAME')] + \
        [ShuffleNetUnitB(24, 240, num_groups=3)] + \
        [ShuffleNetUnitA(240, 240, num_groups=3) for i in range(3)] + \
        [ShuffleNetUnitB(240, 480, num_groups=3)] + \
        [ShuffleNetUnitA(480, 480, num_groups=3) for i in range(7)] + \
        [ShuffleNetUnitB(480, 960, num_groups=3)] + \
        [ShuffleNetUnitA(960, 960, num_groups=3) for i in range(3)] + \
        [AvgPool((7, 7))] + \
        [Flatten] + \
        [Dense(num_classes)]
    
    return stax.serial(*layers)

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
    height = 224
    width = 224
    num_channels = 3
    num_classes = 1000

    key, data_key, labels_key = random.split(key, 3)
    data_shape = (batch_size, height, width, num_channels)
    labels_shape = (batch_size, num_classes)
    data = random.normal(data_key, data_shape)
    labels = random.normal(labels_key, labels_shape)

    # Initialize the network and perform a forward pass
    init_fun, apply_ShuffleNet = ShuffleNet()
    _, params = init_fun(key, data_shape)

    step_size = 0.002
    opt_init, opt_update, get_params = optimizers.sgd(step_size)
    opt_state = opt_init(params)

    @jit
    def update(i, opt_state, inputs, targets):
        params = get_params(opt_state)
        return opt_update(i, grad(cross_entropy_loss)(params, inputs, targets, apply_ShuffleNet), opt_state)

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

    logits = jit(apply_ShuffleNet)(params, data)

    for i in range(number):
        records = []
        for j in range(repeats):
            start_time = time.time()
            logits = jit(apply_ShuffleNet)(params, data)
            end_time = time.time()
            records.append(end_time - start_time)
        print("Average inference latency {} ms".format(1000. * onp.mean(records)))
        print("Median inference latency {} ms".format(1000. * onp.median(records)))

    print(jax.local_devices())