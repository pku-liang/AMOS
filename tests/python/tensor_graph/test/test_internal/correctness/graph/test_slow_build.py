import tvm

from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, ForwardGraph, \
                              BackwardGraph, GraphMutator, PyTIRGraph
import tvm.tensor_graph.nn.functional as F
from tvm.tensor_graph.nn.modules.loss import CELoss
from tvm.tensor_graph.nn.modules.optimize import SGD
from tvm.tensor_graph.core.utils import flatten_tir_graph

ERROR_Conv33 = False

def SimpleNet():
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

    conv29_weight = GraphTensor([128, 64, 31, 31])
    conv29 = F.conv2d_nchw(relu28, conv29_weight, bias=None,\
        stride=1, padding=1)
    # (4, 128, 28, 28)

    alpha30 = GraphTensor([128], name="alpha30")
    beta30 = GraphTensor([128], name="beta30")
    bn30 = F.batch_norm2d(conv29, alpha30, beta30)

    relu32 = F.ReLU(bn30)
    #(4, 128, 28, 28)

    if ERROR_Conv33 == True:
        conv33_weight = GraphTensor([128, 128, 3, 3])
        conv33 = F.conv2d_nchw(relu32, conv33_weight, bias=None,\
            stride=1, padding=1)
        
        pool = F.global_avg_pool2d(conv33)
        #(4, 128, 1, 1)
    else:
        pool = F.global_avg_pool2d(relu32)
        
    print(pool.shape) #(4, 128, 1, 1)
    flatten = F.batch_flatten(pool)
    #(4, 128)

    dense_weight = GraphTensor([1000, 128])
    dense_bias = GraphTensor([1000])
    dense_out = F.dense(flatten, dense_weight, dense_bias)

    label = GraphTensor([4, 1000])

    ce_loss = CELoss(label)
    sgd = SGD(0.002)

    if ERROR_Conv33 == True:
        weights = [alpha0, beta0, conv2_weight, alpha3, beta3, alpha7, beta7, 
            conv10_weight, alpha11, beta11, conv14_weight, conv15_weight, alpha17, 
            beta17, conv20_weight, alpha21, beta21, conv24_weight, alpha26, beta26, 
            conv29_weight, alpha30, beta30, conv33_weight,
            dense_weight, dense_bias]
    else:
        weights = [alpha0, beta0, conv2_weight, alpha3, beta3, alpha7, beta7, 
            conv10_weight, alpha11, beta11, conv14_weight, conv15_weight, alpha17, 
            beta17, conv20_weight, alpha21, beta21, conv24_weight, alpha26, beta26, 
            conv29_weight, alpha30, beta30, #conv33_weight,
            dense_weight, dense_bias]
    
    fgraph = ForwardGraph([data], [dense_out], weights)

    bgraph = fgraph.make_backward(ce_loss, sgd)
    sch, bufs = bgraph.create_schedule()
    print(tvm.lower(sch, bufs, simple_mode=True))

    root_ops = bgraph.outputs + [bgraph.loss] + bgraph.gradients + bgraph.updates
    root_ops = [x.tvm_tensor.op for x in root_ops]
    op_list, _ = flatten_tir_graph(root_ops)
    print(len(op_list))


    target = "llvm"
    dev = 0
    naive_func = bgraph.build(sch, bufs, target)

    
if __name__ == "__main__":
    SimpleNet()