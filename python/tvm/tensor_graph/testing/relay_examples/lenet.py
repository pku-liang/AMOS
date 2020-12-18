import tvm
import numpy as np
from tvm import relay
from tvm.relay.testing import run_infer_type, gradient

def get_lenet(batch_size,
            num_classes=10,
            image_shape=(1, 28, 28),
            dtype="float32"):
    """Get lenet funciton

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of claseses

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    Returns
    -------
    net : relay.Function
        The dataflow.
    """
    data_shape = (batch_size,) + image_shape
    data = relay.TensorType(data_shape, dtype=dtype)
    data = relay.var("data", data)
    conv_w1 = relay.var('c1.weight')
    c1 = relay.nn.conv2d(data=data, weight=conv_w1, channels=6, kernel_size=(5, 5),
                         strides=(1, 1), padding=(2, 2))
    conv_b1 = relay.var('c1.bias', dtype=dtype)
    c1 = relay.nn.bias_add(c1, conv_b1, axis=-1)
    act_c1 = relay.nn.relu(data=c1)
    # Max-pooling
    # [64, 6, 14, 14]
    conv_w2 = relay.var('c2.weight', dtype=dtype)
    conv_b2 = relay.var('c2.bias', dtype=dtype)
    p1 = relay.nn.conv2d(data=act_c1, weight=conv_w2, channels=6, kernel_size=(2, 2),
                         strides=(2, 2), padding=(0, 0))
    p1 = relay.nn.bias_add(p1, conv_b2, axis=-1)
    # Convolution
    conv_w3 = relay.var('c3.weight', dtype=dtype)
    conv_b3 = relay.var('c3.bias', dtype=dtype)
    c2 = relay.nn.conv2d(data=p1, weight=conv_w3, channels=6, kernel_size=(5, 5),
                         strides=(1, 1), padding=(0, 0))
    c2 = relay.nn.bias_add(c2, conv_b3, axis=-1)
    # [64, 6, 28, 28]conv2d(p1, 16, (5, 5), (1, 1), (0, 0), 'c2')  # [64, 16, 10, 10]
    act_c2 = relay.nn.relu(data=c2)
    # Max-pooling
    # [64, 16, 5, 5]
    conv_w4 = relay.var('c4.weight', dtype=dtype)
    conv_b4 = relay.var('c4.bias', dtype=dtype)
    p2 = relay.nn.conv2d(data=act_c2, weight=conv_w4, channels=6, kernel_size=(2, 2),
                          strides=(2, 2), padding=(0, 0))
    p2 = relay.nn.bias_add(p2, conv_b4, axis=-1)
    # reshape
    r1 = relay.nn.batch_flatten(data=p2)
    w1 = relay.var('fc1.weight', dtype=dtype)
    b1 = relay.var('fc1.bias', dtype=dtype)
    fc1 = relay.nn.dense(data=r1, weight=w1, units=128)
    fc1 = relay.nn.bias_add(fc1, b1, axis=-1)
    act1 = relay.nn.relu(data=fc1)
    w2 = relay.var('fc2.weight', dtype=dtype)
    b2 = relay.var('fc2.bias', dtype=dtype)
    fc2 = relay.nn.dense(data=act1, weight=w2, units=64)
    fc2 = relay.nn.bias_add(fc2, b2, axis=-1)
    act2 = relay.nn.relu(data=fc2)
    w3 = relay.var('fc3.weight', dtype=dtype)
    b3 = relay.var('fc3.bias', dtype=dtype)
    fc3 = relay.nn.dense(data=act2, weight=w3, units=num_classes)
    fc3 = relay.nn.bias_add(fc3, b3, axis=-1)
    lenet = relay.nn.softmax(data=fc3)
    argu_list = [conv_w1, conv_b1, conv_w2, conv_b2, w1, b1, w2, b2, w3, b3]
    return relay.Function(relay.analysis.free_vars(lenet), lenet), argu_list


def make_sgd_update_net(loss_function, var, lr=0.002, scale=1.0, wd=0.0, clip=None):
    type_loss_function = run_infer_type(loss_function)
    grad_func = run_infer_type(gradient(type_loss_function))
    grads = relay.TupleWrapper(relay.TupleGetItem(grad_func.body, 1), len(loss_function.params))
    useful_grad = []
    type_var = []
    for var_item in var:
        for index, value_item in enumerate(type_loss_function.params):
            if var_item.name_hint == value_item.name_hint:
                useful_grad.append(grads[index])
                type_var.append(value_item)
                break
        else:
            raise("can't get required params from loss function, internal error")
    updates = []
    for i, v in enumerate(type_var):
        g = useful_grad[i]
        g = relay.multiply(g, relay.const(scale, "float32"))
        if clip is not None:
            g = relay.clip(g, a_min=-1 * clip, a_max=clip)
        g = relay.subtract(v, 
                           relay.multiply(relay.const(lr, "float32"), 
                                          relay.add(g, 
                                                    relay.multiply(relay.const(wd, "float32"), 
                                                                   v))))
        updates.append(g)
    sgd_body = relay.Tuple(updates)
    return relay.Function(relay.analysis.free_vars(sgd_body), sgd_body)


def make_adam_update_net(loss_function, var, lr=0.001, beta1=0.9, beta2=0.99, scale=1.0, wd=0.0, clip=None, name="adam", dtype='float32'):
    type_loss_function = run_infer_type(loss_function)
    grad_func = run_infer_type(gradient(type_loss_function))
    grads = relay.TupleWrapper(relay.TupleGetItem(grad_func.body, 1), len(loss_function.params))
    useful_grad = []
    type_var = []
    for var_item in var:
        for index, value_item in enumerate(type_loss_function.params):
            if var_item.name_hint == value_item.name_hint:
                useful_grad.append(grads[index])
                type_var.append(value_item)
                break
        else:
            raise("can't get required params from loss function, internal error")
    print(type_var)
    updates = []
    m = []
    t = relay.zeros(shape=[1], dtype=dtype)
    epsilon = 1e-04
    const_1 = relay.const(1, dtype=dtype)
    const_beta1 = relay.const(beta1, dtype=dtype)
    const_beta2 = relay.const(beta2, dtype=dtype)
    for i, va in enumerate(type_var):
        m.append(relay.zeros_like(va))
    update_t = relay.add(t, const_1)
    rate = relay.divide(relay.sqrt(relay.subtract(const_1, relay.power(const_beta2, update_t))),
                        relay.subtract(const_1, relay.power(const_beta1, update_t)))
    lr_t = relay.multiply(relay.const(lr, dtype=dtype), rate)
    for var, g, m in zip(type_var, useful_grad, m):
        update_m = relay.add(relay.multiply(const_beta1, m), 
                             relay.multiply(relay.subtract(const_1, const_beta1), g))
        update_v = relay.add(relay.multiply(const_beta2, m), 
                             relay.multiply(relay.subtract(const_1, const_beta2), 
                                            relay.multiply(g, g)))
        update_var = relay.subtract(var, 
                                    relay.divide(relay.multiply(lr_t, update_m), 
                                                 relay.add(relay.sqrt(update_v), 
                                                           relay.const(epsilon, dtype="float32"))))
        updates.append(update_var)
    adam_body = relay.Tuple(updates)
    return relay.Function(relay.analysis.free_vars(adam_body), adam_body)


def mse_loss(lenet_function, target):
    sub = relay.subtract(lenet_function.body, target)
    loss_body = relay.sum(relay.multiply(sub, sub))
    return relay.Function(relay.analysis.free_vars(loss_body), loss_body)
    # return sum((predict - target)**2) / 2.0


def cross_entropy_loss(lenet_function, target):
    loss_body = relay.negative(relay.sum(relay.multiply(relay.log(relay.add(lenet_function.body, 
                                                                            relay.const(1e-5, dtype="float32"))), 
                                                                  target)))
    return relay.Function(relay.analysis.free_vars(loss_body), loss_body)


def make_loss_net(lenet_function, target, optim="CROSS"):
    """Get loss funtion for lenet

    Parameters
    ----------
    lenet_function : relay.Function

    target : relay.Expr

    optim : str, optional
        loss_function strategy, "CROSS" or "MSE"

    Returns
    -------
    net : relay.Function
        The dataflow.
    """
    if optim == "CROSS":
        return cross_entropy_loss(lenet_function, target)
    if optim == "MSE":
        return mse_loss(lenet_function, target)
    raise("unknown optim, use 'CROSS' or 'MSE'.")


def make_grad_net(loss_function):
    """Get updated funtion for lenet

    Parameters
    ----------
    loss_function : relay.Function

    Returns
    -------
    net : relay.Function
        The dataflow.
    """
    type_loss_function = run_infer_type(loss_function)
    grad_func = run_infer_type(gradient(type_loss_function))
    return grad_func


def make_update_net(loss_function, weights, optim="SGD"):
    """Get updated funtion for lenet

    Parameters
    ----------
    loss_function : relay.Function

    weights : [relay.var]
        vars to compute gradient

    optim : str, optional
        updated_function strategy, "ADAM" or "SGD"

    Returns
    -------
    net : relay.Function
        The dataflow.
    """
    if optim == "ADAM":
        return make_adam_update_net(loss_function, weights)
    if optim == "SGD":
        return make_sgd_update_net(loss_function, weights)
    raise("unknown optim, use 'ADAM' or 'SGD'.")


def create_workload(net, initializer=None, seed=0):
    """Helper function to create benchmark image classification workload.

    Parameters
    ----------
    net : tvm.relay.Function
        The selected function of the network.

    initializer : Initializer
        The initializer used

    seed : int
        The seed used in initialization.

    Returns
    -------
    mod : tvm.IRModule
        The created relay module.

    params : dict of str to NDArray
        The parameters.
    """
    mod = tvm.IRModule.from_expr(net)
    mod = relay.transform.InferType()(mod)
    shape_dict = {
        v.name_hint : v.checked_type for v in mod["main"].params}
    np.random.seed(seed)
    initializer = initializer if initializer else Xavier()
    params = {}
    for k, v in shape_dict.items():
        # modify here, skip "label" as well
        if k == "data" or k == "label":
            continue
        init_value = np.zeros(v.concrete_shape).astype(v.dtype)
        initializer(k, init_value)
        params[k] = tvm.nd.array(init_value, ctx=tvm.cpu(0))
    return mod, params
