import tvm
import numpy as np
from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, ForwardGraph, \
                              BackwardGraph, GraphMutator, PyTIRGraph
import tvm.tensor_graph.nn.functional as F
from tvm.tensor_graph.nn.modules.loss import CELoss
from tvm.tensor_graph.nn.modules.optimize import SGD

batch_size = 4
num_class = 1000
channel = 3
height = width = 224
image_shape = (channel, height, width)
data_shape = (batch_size, ) + image_shape
out_shape = (batch_size, num_class)

def resnet():
    data = GraphTensor([4, 3, 224, 224], name="data") #(4, 3, 224, 224)

    alpha0 = GraphTensor([3], name="alpha0")
    beta0 = GraphTensor([3], name="beta0")
    bn0 = F.batch_norm2d(data, alpha0, beta0)

    conv2_weight = GraphTensor([64, 3, 7, 7])
    conv2 = F.conv2d_nchw(bn0, conv2_weight, bias=None, \
        stride=2, padding=3)
    #(4, 64, 112, 112), %2

    alpha3 = GraphTensor([64], name="alpha3")
    beta3 = GraphTensor([64], name="beta3")
    bn3 = F.batch_norm2d(conv2, alpha3, beta3)

    relu5 = F.ReLU(bn3)
    # %5

    maxpool6 = F.avgpool2d(relu5, kernel_h=3, kernel_w=3, \
        stride_h=2, stride_w=2, padding=1)
    #(4, 64, 56, 56)

    alpha7 = GraphTensor([64], name="alpha7")
    beta7 = GraphTensor([64], name="beta7")
    bn7 = F.batch_norm2d(maxpool6, alpha7, beta7)

    relu9 = F.ReLU(bn7)
    # %9
    
    conv10_weight = GraphTensor([64, 64, 3, 3])
    conv10 = F.conv2d_nchw(relu9, conv10_weight, bias=None,\
        stride=1, padding=1)
    #(4, 64, 56, 56)

    alpha11 = GraphTensor([64], name="alpha11")
    beta11 = GraphTensor([64], name="beta11")
    bn11 = F.batch_norm2d(conv10, alpha11, beta11)

    relu13 = F.ReLU(bn11)
    # %13

    conv14_weight = GraphTensor([64, 64, 3, 3])
    conv14 = F.conv2d_nchw(relu13, conv14_weight, bias=None,\
        stride=1, padding=1)
    #(4, 64, 56, 56), %14

    conv15_weight = GraphTensor([64, 64, 1, 1])
    conv15 = F.conv2d_nchw(relu9, conv15_weight, bias=None,\
        stride=1, padding=0)
    #relu11 is far above!
    #(4, 64, 56, 56), %15

    add16 = F.add_4d(conv14, conv15) #Skip Connection
    # %16


    alpha17 = GraphTensor([64], name="alpha17")
    beta17 = GraphTensor([64], name="beta17")
    bn17 = F.batch_norm2d(add16, alpha17, beta17)

    relu19 = F.ReLU(bn17)
    # %19

    conv20_weight = GraphTensor([64, 64, 3, 3])
    conv20 = F.conv2d_nchw(relu19, conv20_weight, bias=None,\
        stride=1, padding=1)
    
    alpha21 = GraphTensor([64], name="alpha21")
    beta21 = GraphTensor([64], name="beta21")
    bn21 = F.batch_norm2d(conv20, alpha21, beta21)

    relu23 = F.ReLU(bn21)
    # %23

    conv24_weight = GraphTensor([64, 64, 3, 3])
    conv24 = F.conv2d_nchw(relu23, conv24_weight, bias=None,\
        stride=1, padding=1)
    
    add25 = F.add_4d(conv24, add16)

    alpha26 = GraphTensor([64], name="alpha26")
    beta26 = GraphTensor([64], name="beta26")
    bn26 = F.batch_norm2d(add25, alpha26, beta26)

    relu28 = F.ReLU(bn26)
    # (4, 64, 56, 56)

    conv29_weight = GraphTensor([128, 64, 3, 3])
    conv29 = F.conv2d_nchw(relu28, conv29_weight, bias=None,\
        stride=2, padding=1)
    # (4, 128, 28, 28)

    alpha30 = GraphTensor([128], name="alpha30")
    beta30 = GraphTensor([128], name="beta30")
    bn30 = F.batch_norm2d(conv29, alpha30, beta30)

    relu32 = F.ReLU(bn30)

    conv33_weight = GraphTensor([128, 128, 3, 3])
    conv33 = F.conv2d_nchw(relu32, conv33_weight, bias=None,\
        stride=1, padding=1)

    conv34_weight = GraphTensor([128, 64, 1, 1])
    conv34 = F.conv2d_nchw(relu28, conv34_weight, bias=None,\
        stride=2, padding=0)

    add35 = F.add_4d(conv33, conv34)

    alpha36 = GraphTensor([128], name="alpha36")
    beta36 = GraphTensor([128], name="beta36")
    bn36 = F.batch_norm2d(add35, alpha36, beta36)

    relu38 = F.ReLU(bn36)

    conv39_weight = GraphTensor([128, 128, 3, 3])
    conv39 = F.conv2d_nchw(relu38, conv39_weight, bias=None,\
        stride=1, padding=1)
    
    alpha40 = GraphTensor([128], name="alpha40")
    beta40 = GraphTensor([128], name="beta40")
    bn40 = F.batch_norm2d(conv39, alpha40, beta40)

    relu42 = F.ReLU(bn40)
    
    conv43_weight = GraphTensor([128, 128, 3, 3])
    conv43 = F.conv2d_nchw(relu42, conv43_weight, bias=None,\
        stride=1, padding=1)
    
    add44 = F.add_4d(conv43, add35)

    alpha45 = GraphTensor([128], name="alpha45")
    beta45 = GraphTensor([128], name="beta45")
    bn45 = F.batch_norm2d(add44, alpha45, beta45)

    relu47 = F.ReLU(bn45)
    #(4, 128, 28, 28)

    conv48_weight = GraphTensor([256, 128, 3, 3])
    conv48 = F.conv2d_nchw(relu47, conv48_weight, bias=None,\
        stride=2, padding=1)
    #(4, 256, 14, 14)

    alpha49 = GraphTensor([256], name="alpha49")
    beta49 = GraphTensor([256], name="beta49")
    bn49 = F.batch_norm2d(conv48, alpha49, beta49)

    relu51 = F.ReLU(bn49)

    conv52_weight = GraphTensor([256, 256, 3, 3])
    conv52 = F.conv2d_nchw(relu51, conv52_weight, bias=None,\
        stride=1, padding=1)
    
    conv53_weight = GraphTensor([256, 128, 1, 1])
    conv53 = F.conv2d_nchw(relu47, conv53_weight, bias=None,\
        stride=2, padding=0)
    #(4, 256, 14, 14)

    add54 = F.add_4d(conv52, conv53)

    alpha55 = GraphTensor([256], name="alpha55")
    beta55 = GraphTensor([256], name="beta55")
    bn55 = F.batch_norm2d(add54, alpha55, beta55)

    relu57 = F.ReLU(bn55)

    conv58_weight = GraphTensor([256, 256, 3, 3])
    conv58 = F.conv2d_nchw(relu57, conv58_weight, bias=None,\
        stride=1, padding=1)

    alpha59 = GraphTensor([256], name="alpha59")
    beta59 = GraphTensor([256], name="beta59")
    bn59 = F.batch_norm2d(conv58, alpha59, beta59)

    relu61 = F.ReLU(bn59)

    conv62_weight = GraphTensor([256, 256, 3, 3])
    conv62 = F.conv2d_nchw(relu61, conv62_weight, bias=None,\
        stride=1, padding=1)
    
    add63 = F.add_4d(conv62, add54)

    alpha64 = GraphTensor([256], name="alpha64")
    beta64 = GraphTensor([256], name="beta64")
    bn64 = F.batch_norm2d(add63, alpha64, beta64)

    relu66 = F.ReLU(bn64)
    #(4, 256, 14, 14)

    conv67_weight = GraphTensor([512, 256, 3, 3])
    conv67 = F.conv2d_nchw(relu66, conv67_weight, bias=None,\
        stride=2, padding=1)
    #(4, 512, 7, 7)

    alpha68 = GraphTensor([512], name="alpha68")
    beta68 = GraphTensor([512], name="beta68")
    bn68 = F.batch_norm2d(conv67, alpha68, beta68)

    relu70 = F.ReLU(bn68)

    conv71_weight = GraphTensor([512, 512, 3, 3])
    conv71 = F.conv2d_nchw(relu70, conv71_weight, bias=None,\
        stride=1, padding=1)
    
    conv72_weight = GraphTensor([512, 256, 1, 1])
    conv72 = F.conv2d_nchw(relu66, conv72_weight, bias=None,\
        stride=2, padding=0)

    add73 = F.add_4d(conv71, conv72)

    alpha74 = GraphTensor([512], name="alpha74")
    beta74 = GraphTensor([512], name="beta74")
    bn74 = F.batch_norm2d(add73, alpha74, beta74)

    relu76 = F.ReLU(bn74)

    conv77_weight = GraphTensor([512, 512, 3, 3])
    conv77 = F.conv2d_nchw(relu76, conv77_weight, bias=None,\
        stride=1, padding=1)

    alpha78 = GraphTensor([512], name="alpha78")
    beta78 = GraphTensor([512], name="beta78")
    bn78 = F.batch_norm2d(conv77, alpha78, beta78)

    relu80 = F.ReLU(bn78)

    conv81_weight = GraphTensor([512, 512, 3, 3])
    conv81 = F.conv2d_nchw(relu80, conv81_weight, bias=None,\
        stride=1, padding=1)
    
    add82 = F.add_4d(conv81, add73)

    alpha83 = GraphTensor([512], name="alph83")
    beta83 = GraphTensor([512], name="beta83")
    bn83 = F.batch_norm2d(add82, alpha83, beta83)

    relu85 = F.ReLU(bn83)

    pool86 = F.global_avg_pool2d(relu85)
    
    flatten87 = F.batch_flatten(pool86)
    #(4, 512)

    dense_weight = GraphTensor([1000, 512])
    dense_bias = GraphTensor([1000])
    dense89 = F.dense(flatten87, dense_weight, dense_bias)
    #(4, 1000)

    label = GraphTensor([4, 1000])
    
    ce_loss = CELoss(label)
    sgd = SGD(0.002)

    weights = [alpha0, beta0, conv2_weight, alpha3, beta3, alpha7, beta7, 
        conv10_weight, alpha11, beta11, conv14_weight, conv15_weight, alpha17, 
        beta17, conv20_weight, alpha21, beta21, conv24_weight, alpha26, beta26, 
        conv29_weight, alpha30, beta30, conv33_weight, conv34_weight, alpha36, 
        beta36, conv39_weight, alpha40, beta40, conv43_weight, alpha45, beta45, 
        conv48_weight, alpha49, beta49, conv52_weight, conv53_weight, alpha55, 
        beta55, conv58_weight, alpha59, beta59, conv62_weight, alpha64, beta64, 
        conv67_weight, alpha68, beta68, conv71_weight, conv72_weight, alpha74, 
        beta74, conv77_weight, alpha78, beta78, conv81_weight, alpha83, beta83, 
        dense_weight, dense_bias]
    fgraph = ForwardGraph([data], [dense89], weights)

    bgraph = fgraph.make_backward(ce_loss, sgd)
    sch, bufs = bgraph.create_schedule()
    #print(tvm.lower(sch, bufs, simple_mode=True))
    target = "llvm"
    dev = 0
    #naive_func = bgraph.build(sch, bufs, target)
    
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
    label_np = np.random.uniform(-1, 1, [4, 1000]).astype("float32")
    tgraph.set_inputs({bgraph.inputs[0].tvm_tensor: A_np})
    tgraph.set_labels({bgraph.labels[0].tvm_tensor: label_np})
    tgraph.allocate_buffer(target, dev)

    # # get golden result
    # ctx = tvm.context(target, dev)
    # # copy the data (do not use reference)
    # A_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.inputs[0].tvm_tensor).asnumpy(), ctx)
    # label_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.labels[0].tvm_tensor).asnumpy(), ctx)
    # B_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[0].tvm_tensor).asnumpy(), ctx)
    # bias_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[1].tvm_tensor).asnumpy(), ctx)
    # E_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[2].tvm_tensor).asnumpy(), ctx)
    # updates_tvm = [tvm.nd.array(x.asnumpy(), ctx) for x in tgraph.get_updates()]
    # naive_func(A_tvm, label_tvm, B_tvm, bias_tvm, E_tvm, *updates_tvm)
    
    # compute target value
    for mark in tgraph.call_order:
        func = tgraph.functions[mark]
        bufs = tgraph.bufs[mark]
        real_bufs = [tgraph.tvm_array_dict[tgraph.subgraphs[mark].index[x]] for x in bufs]
        func(*real_bufs)
    print("Checking Gradients!")
    for item in tgraph.get_updates():
        print(item.asnumpy())

    # # test correctness
    # for (gold, value) in zip(updates_tvm, tgraph.get_updates()):
    #     tvm.testing.assert_allclose(gold.asnumpy(), value.asnumpy(), atol=1e-5, rtol=1e-30)

    # print("Success!")


if __name__ == "__main__":
    resnet()