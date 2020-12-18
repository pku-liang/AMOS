import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from tensor_graph.nn.layers import Layer, Conv2d, BatchNorm2d, ReLU, \
                                  AvgPool2d, GlobalAvgPool2d, Linear, Sequential, CapsuleConv2d
from tensor_graph.nn.functional import elementwise_add
import tvm
from tensor_graph.core import compute, GraphTensor, GraphOp, GraphNode

from tensor_graph.testing.models import helper

batch_size = 20
capsule_nums = 8

class ConvLayer(Layer):
    
    def __init__(self, in_channels=1, out_channels=256):
        '''Constructs the ConvLayer with a specified input and output size.
           param in_channels: input depth of an image, default value = 1
           param out_channels: output depth of the convolutional layer, default value = 256
           '''
        super(ConvLayer, self).__init__()

        # defining a convolutional layer of the specified size
        self.conv = Conv2d(in_channels, out_channels, kernel_size=9, bias=True, stride=1, padding=0)
        self.relu = ReLU()

    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input to the layer; an input image
           return: a relu-activated, convolutional layer
           '''
        # x torch.Size([20, 1, 28, 28])
        conved = self.conv(x)
        # conved torch.Size([20, 256, 20, 20])
        features = self.relu(conved)
        # features torch.Size([20, 256, 20, 20])
        return features


def concat_eight_vector_lastdim(cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7):
    batch, channel, h, w, _ = cap0.shape
    assert _ == 1
    eight = 8
    def _inner_cat(batch, channel, h, w, eight, cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7, requires_grad=True):
        return compute([batch, channel, h, w, eight],
                lambda i, j, p, q, k:
                    tvm.te.if_then_else(k == 0, cap0[i, j, p, q, k],
                        tvm.te.if_then_else(k == 1, cap1[i, j, p, q, k-1],
                            tvm.te.if_then_else(k == 2, cap2[i, j, p, q, k-2],
                                tvm.te.if_then_else(k == 3, cap3[i, j, p, q, k-3],
                                    tvm.te.if_then_else(k == 4, cap4[i, j, p, q, k-4],
                                        tvm.te.if_then_else(k == 5, cap5[i, j, p, q, k-5], 
                                            tvm.te.if_then_else(k == 6, cap6[i, j, p, q, k-6],
                                                cap7[i, j, p, q, k-7]))))))),
                name="concat8",
                tag="concat8",
                requires_grad=requires_grad)
    return GraphOp([batch, channel, h, w, eight], [], 
                    [cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7],
                    _inner_cat, name="concat8")

class PrimaryCaps(Layer):
    
    def __init__(self, num_capsules=capsule_nums, in_channels=256, out_channels=32):
        '''Constructs a list of convolutional layers to be used in 
           creating capsule output vectors.
           param num_capsules: number of capsules to create
           param in_channels: input depth of features, default value = 256
           param out_channels: output depth of the convolutional layers, default value = 32
           '''
        super(PrimaryCaps, self).__init__()
        self.num_caps = num_capsules
        self.conv0 = CapsuleConv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0, num_caps=1)
        self.conv1 = CapsuleConv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0, num_caps=1)
        self.conv2 = CapsuleConv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0, num_caps=1)
        self.conv3 = CapsuleConv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0, num_caps=1)
        self.conv4 = CapsuleConv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0, num_caps=1)
        self.conv5 = CapsuleConv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0, num_caps=1)
        self.conv6 = CapsuleConv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0, num_caps=1)
        self.conv7 = CapsuleConv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0, num_caps=1)
        # self.capsules = CapsuleConv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0, num_caps=num_capsules)

    
    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input; features from a convolutional layer
           return: a set of normalized, capsule output vectors
           '''
        # get batch size of inputs
        # x torch.Size([20, 256, 20, 20])
        # reshape convolutional layer outputs to be (batch_size, vector_dim=1152, 1)
        conved0 = self.conv0(x)
        conved1 = self.conv1(x)
        conved2 = self.conv2(x)
        conved3 = self.conv3(x)
        conved4 = self.conv4(x)
        conved5 = self.conv5(x)
        conved6 = self.conv6(x)
        conved7 = self.conv7(x)
        cap_u = concat_eight_vector_lastdim(conved0, conved1, conved2, conved3, conved4, conved5, conved6, conved7)
        
        # cap_u = self.capsules(x)

        # cap_u [20, 32, 6, 6, num_caps]

        u_cat = helper.two_flatten(cap_u, self.num_caps)
        # u_cat [20, 32 * 6 * 6, num_caps]

        # for capsule in self.capsules:
        #     add_into_u = capsule(x)
        #     # add_into_u torch.Size([20, 32, 6, 6])
        #     viewed = helper.two_flatten(add_into_u)
        #     # viewed = add_into_u.view(batch_size, 32 * 6 * 6, 1)
        #     u.append(viewed)
        # u = [capsule(x).view(batch_size, 32 * 6 * 6, 1) for capsule in self.capsules]
        # stack up output vectors, u, one for each capsule
        # u_cat = helper.concat_eight_vector_lastdim(*u)
        # u = torch.cat(u, dim=-1)
        # (batch_size, 32*6*6, num_capsules)
        # squashing the stack of vectors

        u_squash = self.squash(u_cat)

        return u_squash

        
    def squash(self, input):
        # input[20, 1152, 8]
        bz, vector_dim, num_capsules = input.shape
        # assert bz == batch_size 
        assert vector_dim == 32*6*6 and num_capsules == self.num_caps
        squared_norm = helper.norm(input, self.num_caps)
        # squared_norm [20, 32 * 6 * 6]
        #scale = helper.scaling(squared_norm)
        # scale [20, 32 * 6 * 6]
        out_squash = helper.weight_squash(input, squared_norm, self.num_caps)

        # integrate scaling into weight_squash
        # out_squash = helper.weight_squash(input, squared_norm)
        return out_squash
        # (batch_size, 32*6*6, num_capsules)



def dynamic_routing(b_ij, u_hat, routing_iterations=3):
    '''Performs dynamic routing between two capsule layers.
       param b_ij: initial log probabilities that capsule i should be coupled to capsule j
       param u_hat: input, weighted capsule vectors, W u
       param routing_iterations: number of times to update coupling coefficients
       return: v_j, output capsule vectors
       '''
       # update b_ij, c_ij for number of routing iterations
       # b_ij torch.Size([10, 20, 1152, 16])
       # u_hat torch.Size([10, 20, 1152, 16])
    # iter 0
    c_ij = helper.softmax_2(b_ij)
    s_j = helper.cu_multiply(c_ij, u_hat)
    v_j = helper.squash2(s_j)
    a_ij = helper.uv_dot(u_hat, v_j)
    b_ij0 = helper.update_by_aij(b_ij, a_ij, "1")

    # iter1
    c_ij1 = helper.softmax_2(b_ij0)
    s_j1 = helper.cu_multiply(c_ij1, u_hat)
    v_j1 = helper.squash2(s_j1)
    a_ij1 = helper.uv_dot(u_hat, v_j1)
    b_ij1 = helper.update_by_aij(b_ij0, a_ij1, "2")

    # iter 2
    c_ij2 = helper.softmax_2(b_ij1)
    s_j2 = helper.cu_multiply(c_ij2, u_hat)
    v_j2 = helper.squash2(s_j2)



    # for iteration in range(routing_iterations):
    #     # softmax calculation of coupling coefficients, c_ij
    #     c_ij = helper.softmax_2(b_ij)
    #     # c_ij torch.Size([10, 20, 1152, 16])

    #     # calculating total capsule inputs, s_j = sum(c_ij*u_hat)
    #     s_j = helper.cu_multiply(c_ij, u_hat)
    #     # s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
    #     # s_j torch.Size([10, 20, 1, 1, 16]) ----> [10, 20, 1, 16] -> [10, 20, 16]

    #     # squashing to get a normalized vector output, v_j
    #     v_j = helper.squash2(s_j)
    #     # v_j  [10, 20, 16]

    #     # if not on the last iteration, calculate agreement and new b_ij
    #     if iteration < routing_iterations - 1:
    #         # agreement
            
    #         # v_j  [10, 20, 16]
    #         # u_hat torch.Size([10, 20, 1152, 16])
    #         a_ij = helper.uv_dot(u_hat, v_j)
    #         # a_ij = (u_hat * v_j).sum(dim=-1, keepdim=false!)
    #         # a_ij torch.Size([10, 20, 1152])
            
    #         # b_ij, ([10, 20, 1152, 16])
    #         b_ij = helper.update_by_aij(b_ij, a_ij)
    #         # b_ij = b_ij + a_ij
    #         # b_ij [10, 20, 1152, 16]
    
    # v_j [10, 20, 16]
    return v_j2 # return latest v_j


class DigitCaps(Layer):
    
    def __init__(self, num_capsules=10, previous_layer_nodes=32*6*6, 
                 in_channels=capsule_nums, out_channels=16):
        '''Constructs an initial weight matrix, W, and sets class variables.
           param num_capsules: number of capsules to create
           param previous_layer_nodes: dimension of input capsule vector, default value = 1152
           param in_channels: number of capsules in previous layer, default value = 8
           param out_channels: dimensions of output capsule vector, default value = 16
           '''
        super(DigitCaps, self).__init__()

        # setting class variables
        self.num_capsules = num_capsules
        self.previous_layer_nodes = previous_layer_nodes # vector input (dim=1152)
        self.in_channels = in_channels # previous layer's number of capsules

        # starting out with a randomly initialized weight matrix, W
        # these will be the weights connecting the PrimaryCaps and DigitCaps layers
        self.W = GraphTensor((num_capsules, previous_layer_nodes, in_channels, out_channels),
                    dtype="float32", name="self.W", requires_grad=True)
        self.b_ij = GraphTensor((num_capsules, batch_size, previous_layer_nodes, out_channels), dtype="float32", name="b_ij", requires_grad=True)
        # self.W = nn.Parameter(torch.randn(num_capsules, previous_layer_nodes, 
        #                                   in_channels, out_channels))

    def forward(self, u):
        '''Defines the feedforward behavior.
           param u: the input; vectors from the previous PrimaryCaps layer
           return: a set of normalized, capsule output vectors
           '''
        
    #    # original_u torch.Size([20, 1152, 8])
    #     u = u[None, :, :, None, :]
    #     #  u torch.Size([1, 20, 1152, 1, 8])

    #     #  original_W torch.Size([10, 1152, 8, 16])
    #     W = self.W[:, None, :, :, :]
    #     # W torch.Size([10, 1, 1152, 8, 16])
        
    #     # calculating u_hat = W*u
    #     u_hat = torch.matmul(u, W)
    #     # u_hat torch.Size([10, 20, 1152, 16])

        # u torch.Size([20, 1152, 8])
        # W  [10, 1152, 8, 16]
        u_hat = helper.uW_multiply(u, self.W, self.in_channels)
        # u_hat [10, 20, 1152, 16]

        # getting the correct size of b_ij
        # setting them all to 0, initially
        #b_ij = torch.zeros(*u_hat.size())
        # b_ij = GraphTensor((10, 20, 1152, 16), dtype="float32", name="b_ij", requires_grad=False)
        # b_ij torch.Size([10, 20, 1152, 16])

        # update coupling coefficients and calculate v_j
        v_j = dynamic_routing(self.b_ij, u_hat, routing_iterations=3)
        # v_j [10, 20, 16]

        return v_j # return final vector outputs

class CapsuleNetwork(Layer):
    
    def __init__(self):
        '''Constructs a complete Capsule Network.'''
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        #self.decoder = Decoder()
                
    def forward(self, images):
        '''Defines the feedforward behavior.
           param images: the original MNIST image input data
           return: output of DigitCaps layer, reconstructed images, class scores
           '''
        batch, n1, n28, n28_ = images.shape
        assert batch == batch_size
        assert n1 == 1 and n28 == 28 and n28_ == 28
        primary_caps_output = self.primary_capsules(self.conv_layer(images))
        caps_output = self.digit_capsules(primary_caps_output)#.squeeze().transpose(0,1)
        # caps_output [10, 20, 1, 1, 16] -> deprecated
        # caps_output2 = helper.squeeze_transpose(caps_output)
        # caps_output2:[20, 10, 16]
        # reconstructions, y = self.decoder(caps_output)

        # [10, 20, 16]
        return caps_output#, reconstructions, y

def get_model(batch=20, num_cap=8):
    global batch_size
    batch_size = batch
    global capsule_nums
    capsule_nums = num_cap
    return CapsuleNetwork()