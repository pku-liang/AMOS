import os

import numpy as np

import tvm
import time
from tvm import te
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
import lenet
import tvm.relay.testing.init as init
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
from tvm.tensor_graph.testing.datasets import load_mnist_dataset
from keras.utils import to_categorical
import tvm.contrib.graph_runtime as runtime


batch_size = 128


def count_hits(predict, label):
    pred = predict.argmax(axis=1)
    acc = (np.array(pred) == np.array(label)).sum()
    return acc


def load_dataset(dataset="mnist.npz"):
    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []
    train_loader, test_loader = load_mnist_dataset(batch_size)
    for v in train_loader:
      train_datas.append(v[0].numpy())
      train_labels.append(v[1].numpy())
    for v in test_loader:
      test_datas.append(v[0].numpy())
      test_labels.append(v[1].numpy())

    return (np.array(train_datas), np.array(train_labels)), (np.array(test_datas), np.array(test_labels))


def shuffle(x, y):
    length = len(x)
    index = list(range(length))
    np.random.shuffle(index)
    return x[index], y[index]


def train(target):
    ctx = tvm.context(target.kind, 0)
    (train_data, train_label), (test_data, test_label) = load_dataset()
    train_size = len(train_data)
    test_size = len(test_data)
    epoch = 20
    img_shape = (1, 28, 28)
    dtype = "float32"

    lenet_function, weights = lenet.get_lenet(batch_size, num_classes=10, image_shape=img_shape, dtype=dtype)
    num_weights = len(weights)
    label = relay.var("label", shape=(batch_size, 10), dtype=dtype)
    loss_function = lenet.make_loss_net(lenet_function, label, optim="MSE")
    updated_function = lenet.make_update_net(loss_function, weights, optim="ADAM")

    lenet_mod, lenet_params = lenet.create_workload(lenet_function, init.Xavier(), seed=int(time.time()))
    loss_mod, loss_params = lenet.create_workload(loss_function, init.Xavier(), seed=int(time.time()))
    updated_mod, updated_params = lenet.create_workload(updated_function, init.Xavier(), seed=int(time.time()))

    lenet_exe = relay.create_executor("graph", mod=lenet_mod, ctx=ctx, target=target).evaluate()
    loss_exe = relay.create_executor("graph", mod=loss_mod, ctx=ctx, target=target).evaluate()
    # only "debug" work
    updated_exe = relay.create_executor("debug", mod=updated_mod, ctx=ctx, target=target).evaluate()
    updated_arg = updated_params

    for ep in range(epoch):
        cur_train_data, cur_train_label = shuffle(train_data, train_label)
        for i in range(train_size):
            inputs_data = cur_train_data[i]
            label_data = to_categorical(
                cur_train_label[i], num_classes=10)
            updated_arg["data"] = tvm.nd.array(inputs_data.astype("float32"), ctx=ctx)
            updated_arg["label"] = tvm.nd.array(label_data.astype("float32"), ctx=ctx)
            updated_rt_params = []
            # sort params
            for t in updated_function.params:
                updated_rt_params.append(updated_arg[t.name_hint])
            updated_out = list(updated_exe(*updated_rt_params))
            # updated
            for k, var in enumerate(weights):
                updated_arg[var.name_hint] = updated_out[k]

            if (i + 1) % 10 == 0:
                loss_rt_params = []
                for t in loss_function.params:
                    loss_rt_params.append(updated_arg[t.name_hint])
                loss_out = loss_exe(*loss_rt_params)
                print("loss:", loss_out)

        print("Test accuracy")
        hits = 0
        for i in range(test_size // batch_size):
            inputs_data = test_data[i * batch_size:(i+1) * batch_size]
            label_data = test_label[i * batch_size:(i+1) * batch_size]
            lenet_rt_params = []
            for t in lenet_function.params:
                if(t.name_hint == "data"):
                    lenet_rt_params.append(tvm.nd.array(inputs_data.astype("float32"), ctx=ctx))
                else:
                    lenet_rt_params.append(updated_arg[t.name_hint])
            lenet_out = lenet_exe(*lenet_rt_params)
            predict = lenet_out.asnumpy()
            hits += count_hits(predict, label_data)
        print("accuracy is:", hits / float(test_size))


def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)
    elif name == 'lenet':
        img_shape = (1, 28, 28)
        num_classes = 10
        lenet_function, weights = lenet.get_lenet(batch_size, num_classes=num_classes, image_shape=img_shape, dtype=dtype)
        num_weights = len(weights)
        label = relay.var("label", shape=(batch_size, num_classes), dtype=dtype)
        loss_function = lenet.make_loss_net(lenet_function, label, optim="MSE")
        grad_function = lenet.make_grad_net(loss_function)
        mod, params = lenet.create_workload(grad_function, init.Xavier(), seed=int(time.time()))
        input_shape = (batch_size, *img_shape)
        output_shape = (batch_size, num_classes)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def tune_and_evaluate(tuning_opt, target):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params)
                                              # ops=(relay.op.get("nn.conv2d"),))
    
    print(len(tasks))
    for task in tasks:
        print(task)

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))



if __name__ == "__main__":
    #### DEVICE CONFIG ####
    target = tvm.target.cuda()

    #### TUNING OPTION ####
    network = 'lenet'
    log_file = "%s.log" % network
    dtype = 'float32'

    tuning_option = {
        'log_filename': log_file,

        'tuner': 'xgb',
        'n_trial': 2000,
        'early_stopping': 600,

        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }
    tune_and_evaluate(tuning_option, target)

    train(target)
