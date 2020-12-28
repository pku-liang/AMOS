import tvm
import torch
import numpy as np
from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, ForwardGraph, \
                              BackwardGraph, GraphMutator, PyTIRGraph
import tensor_graph.nn.functional as F
from tvm.tensor_graph.nn.modules.loss import CELoss, MSELoss
from tvm.tensor_graph.nn.modules.optimize import SGD

NO_BN = True

def SimpleNet():
    data = GraphTensor([4, 3, 224, 224], name="data") #(4, 3, 224, 224)
    if NO_BN == False:
        alpha0 = GraphTensor([3], name="alpha0")
        beta0 = GraphTensor([3], name="beta0")
        bn0 = F.batch_norm2d(data, alpha0, beta0)

        conv2_weight = GraphTensor([64, 3, 7, 7])
        conv2 = F.conv2d_nchw(bn0, conv2_weight, bias=None, \
            stride=2, padding=3)
        #(4, 64, 112, 112), %2
    else:
        conv2_weight = GraphTensor([64, 3, 7, 7])
        conv2 = F.conv2d_nchw(data, conv2_weight, bias=None, \
            stride=2, padding=3)
        #(4, 64, 112, 112), %2
    relu5 = F.ReLU(conv2)
    # %5
    maxpool6 = F.avgpool2d(relu5, kernel_h=3, kernel_w=3, \
        stride_h=2, stride_w=2, padding=1)
    #(4, 64, 56, 56)

    pool = F.global_avg_pool2d(maxpool6)
    #(4, 64, 1, 1)

    flatten = F.batch_flatten(pool)
    #(4, 64)

    dense_weight = GraphTensor([10, 64])
    dense_bias = GraphTensor([10])
    dense_out = F.dense(flatten, dense_weight, dense_bias)

    label = GraphTensor([4, 10])

    ce_loss = MSELoss(label)
    sgd = SGD(0.01)
    
    if NO_BN == False:
        weights = [alpha0, beta0, conv2_weight, dense_weight, dense_bias]
    else:
        weights = [conv2_weight, dense_weight, dense_bias]
    
    fgraph = ForwardGraph([data], [dense_out], weights)

    bgraph = fgraph.make_backward(ce_loss, sgd)
    sch, bufs = bgraph.create_schedule()
    #print(tvm.lower(sch, bufs, simple_mode=True))

    target = "llvm"
    dev = 0
    naive_func = bgraph.build(sch, bufs, target)

    tgraph = PyTIRGraph(
        [x.tvm_tensor for x in bgraph.inputs],
        [x.tvm_tensor for x in bgraph.labels],
        [x.tvm_tensor for x in bgraph.outputs],
        [x.tvm_tensor for x in bgraph.weights],
        bgraph.loss.tvm_tensor,
        [x.tvm_tensor for x in bgraph.gradients],
        [x.tvm_tensor for x in bgraph.updates])
    
    # apply config
    # 1. modify op stat list -> head, tail
    # 2. make subgraphs
    tgraph.partition_graph()

    # 3. create schedule
    tgraph.create_schedule()
    # 4. modify schedule
    tgraph.build(target)
    # allocate buffer
    # only the first call has effect
    A_np = np.random.uniform(-1, 1, [4, 3, 224, 224]).astype("float32")
    label_np = np.random.uniform(-1, 1, [4, 10]).astype("float32")
    tgraph.set_inputs({bgraph.inputs[0].tvm_tensor: A_np})
    tgraph.set_labels({bgraph.labels[0].tvm_tensor: label_np})
    tgraph.allocate_buffer(target, dev)

    # get golden result
    ctx = tvm.context(target, dev)
    # copy the data (do not use reference)
    A_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.inputs[0].tvm_tensor).asnumpy(), ctx)
    label_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.labels[0].tvm_tensor).asnumpy(), ctx)
    if NO_BN == False:
        alpha_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[0].tvm_tensor).asnumpy(), ctx)
        beta_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[1].tvm_tensor).asnumpy(), ctx)
        conv2_weight_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[2].tvm_tensor).asnumpy(), ctx)
        dense_weight_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[3].tvm_tensor).asnumpy(), ctx)
        dense_bias_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[4].tvm_tensor).asnumpy(), ctx)
        updates_tvm = [tvm.nd.array(x.asnumpy(), ctx) for x in tgraph.get_updates()]
    else:
        #alpha_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[0].tvm_tensor).asnumpy(), ctx)
        #beta_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[1].tvm_tensor).asnumpy(), ctx)
        conv2_weight_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[0].tvm_tensor).asnumpy(), ctx)
        dense_weight_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[1].tvm_tensor).asnumpy(), ctx)
        dense_bias_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[2].tvm_tensor).asnumpy(), ctx)
        updates_tvm = [tvm.nd.array(x.asnumpy(), ctx) for x in tgraph.get_updates()]
    print("before doing anything!!")
    print("dense_bias_tvm", dense_bias_tvm)
    print("get_gradients has", len(tgraph.get_gradients()))
    print(tgraph.get_gradients()[-1].shape, tgraph.get_gradients()[-1])
    print("Checking Updates for first time!")
    print(tgraph.get_updates()[-1].shape, tgraph.get_updates()[-1])
    print("Doing things!!")

    #Here add pytorch golden for comparison
    A_torch = torch.tensor(A_tvm.asnumpy(), requires_grad=True)
    if NO_BN == False:
        alpha_torch = torch.nn.Parameter(torch.tensor(alpha_tvm.asnumpy(), requires_grad=True))
        beta_torch = torch.nn.Parameter(torch.tensor(beta_tvm.asnumpy(),  requires_grad=True))
    conv2_weight_torch = torch.nn.Parameter(torch.tensor(conv2_weight_tvm.asnumpy(), requires_grad=True))
    dense_weight_torch = torch.nn.Parameter(torch.tensor(dense_weight_tvm.asnumpy(), requires_grad=True))
    dense_bias_torch = torch.nn.Parameter(torch.tensor(dense_bias_tvm.asnumpy(), requires_grad=True))
    label_torch = torch.tensor(label_tvm.asnumpy(), requires_grad=False)

    if NO_BN == False:
        bn_torch_layer = torch.nn.BatchNorm2d(3, eps=1e-05, momentum=1, track_running_stats=False)
        #Setting track_running_stats=False, to use the batch statistics
        bn_torch_layer.weight = alpha_torch
        bn_torch_layer.bias = beta_torch

    conv2_torch_layer = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3, dilation=1, groups=1, bias=False)
    relu_torch_layer = torch.nn.ReLU()
    avgpool_torch_layer = torch.nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
    globalpool_torch_layer = torch.nn.AvgPool2d(kernel_size=56)
    loss_torch_layer = torch.nn.MSELoss(reduction="sum")

    if NO_BN == False:
        bn_torch = bn_torch_layer(A_torch)
    else:
        bn_torch = A_torch
    conv_torch = conv2_torch_layer(bn_torch)
    Relu_torch = relu_torch_layer(conv_torch)
    avgpool_torch = avgpool_torch_layer(Relu_torch)
    globalpool_torch = globalpool_torch_layer(avgpool_torch)
    flatten_torch = globalpool_torch.view(4, 64)

    dense_out_torch = torch.nn.functional.linear(flatten_torch, dense_weight_torch, dense_bias_torch)
    loss_torch = loss_torch_layer(dense_out_torch, label_torch)
    loss_torch.backward()
    #Pytorch model is completed!

    # if NO_BN == False:
    #     naive_func(A_tvm, label_tvm, alpha_tvm, beta_tvm, conv2_weight_tvm, 
    #         dense_weight_tvm, dense_bias_tvm, *updates_tvm)
    # else:
    #     naive_func(A_tvm, label_tvm, conv2_weight_tvm, 
    #         dense_weight_tvm, dense_bias_tvm, *updates_tvm)

    
    # compute target value
    print("---------------------------------")
    for mark in tgraph.call_order:
        func = tgraph.functions[mark]
        bufs = tgraph.bufs[mark]
        real_bufs = [tgraph.tvm_array_dict[tgraph.subgraphs[mark].index[x]] for x in bufs]
        func(*real_bufs)

    # test correctness
    # for (gold, value) in zip(updates_tvm, tgraph.get_updates()):
    #     tvm.testing.assert_allclose(gold.asnumpy(), value.asnumpy(), atol=1e-30, rtol=1e-30)

    
    print("After doing things, get_gradients has", len(tgraph.get_gradients()))
    print(tgraph.get_gradients()[-1].shape, tgraph.get_gradients()[-1])
    print("Checking Updates for second time!")
    print(tgraph.get_updates()[-1].shape, tgraph.get_updates()[-1])
    print("")
    if NO_BN == False:
        print("alpha_torch_grad", alpha_torch.grad.numpy())
        print("beta_torch_grad", beta_torch.grad.numpy())
    print("dense_bias_torch_grad", dense_bias_torch.grad.numpy())

    # print("Success!")

    
if __name__ == "__main__":
    SimpleNet()