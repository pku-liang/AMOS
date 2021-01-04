import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from tvm.tensor_graph.nn.layers import Layer, Conv2d, BatchNorm2d, ReLU, \
                                  AvgPool2d, GlobalAvgPool2d, Linear, Sequential, CapsuleConv2d
from tvm.tensor_graph.nn.functional import elementwise_add
from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, GraphNode

from tvm.tensor_graph.testing.models import helper

# batch_size = 20
capsule_nums = 8

class ConvLayer(Layer):
    
    def __init__(self, in_channels=1, out_channels=256, dtype="float32", out_dtype="float32"):
        '''Constructs the ConvLayer with a specified input and output size.
           param in_channels: input depth of an image, default value = 1
           param out_channels: output depth of the convolutional layer, default value = 256
           '''
        super(ConvLayer, self).__init__()

        # defining a convolutional layer of the specified size
        self.conv = Conv2d(in_channels, out_channels, kernel_size=9, bias=True,
            stride=1, padding=0, dtype=dtype, out_dtype=out_dtype)
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

class PrimaryCaps(Layer):
    
    def __init__(self, num_capsules=capsule_nums, in_channels=256, out_channels=32, dtype="float32", out_dtype="float32"):
        '''Constructs a list of convolutional layers to be used in 
           creating capsule output vectors.
           param num_capsules: number of capsules to create
           param in_channels: input depth of features, default value = 256
           param out_channels: output depth of the convolutional layers, default value = 32
           '''
        super(PrimaryCaps, self).__init__()
        self.num_caps = num_capsules
        self.capsules = CapsuleConv2d(
            in_channel=in_channels, out_channel=out_channels,
            kernel_size=9, stride=2, padding=0, num_caps=num_capsules,
            dtype=dtype, out_dtype=out_dtype)

    
    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input; features from a convolutional layer
           return: a set of normalized, capsule output vectors
           '''
        # get batch size of inputs
        # x torch.Size([20, 256, 20, 20])
        # reshape convolutional layer outputs to be (batch_size, vector_dim=1152, 1)

        cap_u = self.capsules(x)
        # cap_u [20, 32, 6, 6, num_caps]

        u_cat = helper.two_flatten(cap_u, self.num_caps)
        # u_cat [20, 32 * 6 * 6, num_caps]

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



def dynamic_routing(b_ij, u_hat, routing_iterations=3, out_dtype="float32"):
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
    s_j = helper.cu_multiply(c_ij, u_hat, out_dtype=out_dtype)
    v_j = helper.squash2(s_j)
    a_ij = helper.uv_dot(u_hat, v_j, out_dtype=out_dtype)
    b_ij0 = helper.update_by_aij(b_ij, a_ij, "1")

    # iter1
    c_ij1 = helper.softmax_2(b_ij0)
    s_j1 = helper.cu_multiply(c_ij1, u_hat, out_dtype=out_dtype)
    v_j1 = helper.squash2(s_j1)
    a_ij1 = helper.uv_dot(u_hat, v_j1, out_dtype=out_dtype)
    b_ij1 = helper.update_by_aij(b_ij0, a_ij1, "2")

    # iter 2
    c_ij2 = helper.softmax_2(b_ij1)
    s_j2 = helper.cu_multiply(c_ij2, u_hat, out_dtype=out_dtype)
    v_j2 = helper.squash2(s_j2)
    
    # v_j [10, 20, 16]
    return v_j2 # return latest v_j


class DigitCaps(Layer):
    
    def __init__(self, batch_size, num_capsules=10, previous_layer_nodes=32*6*6, 
                 in_channels=capsule_nums, out_channels=16, dtype="float32", out_dtype="float32"):
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
        self.out_dtype = out_dtype

        # starting out with a randomly initialized weight matrix, W
        # these will be the weights connecting the PrimaryCaps and DigitCaps layers
        self.W = GraphTensor((num_capsules, previous_layer_nodes, in_channels, out_channels),
                    dtype=dtype, name="self.W", requires_grad=True)
        self.b_ij = GraphTensor((num_capsules, batch_size, previous_layer_nodes, out_channels), dtype=dtype, name="b_ij", requires_grad=True)
        # self.W = nn.Parameter(torch.randn(num_capsules, previous_layer_nodes, 
        #                                   in_channels, out_channels))

    def forward(self, u):
        '''Defines the feedforward behavior.
           param u: the input; vectors from the previous PrimaryCaps layer
           return: a set of normalized, capsule output vectors
           '''

        # u torch.Size([20, 1152, 8])
        # W  [10, 1152, 8, 16]
        u_hat = helper.uW_multiply(u, self.W, self.in_channels, out_dtype=self.out_dtype)
        # u_hat [10, 20, 1152, 16]

        # update coupling coefficients and calculate v_j
        v_j = dynamic_routing(self.b_ij, u_hat, routing_iterations=3, out_dtype=self.out_dtype)
        # v_j [10, 20, 16]

        return v_j # return final vector outputs

class CapsuleNetwork(Layer):
    
    def __init__(self, batch_size, dtype="float32", out_dtype="float32"):
        '''Constructs a complete Capsule Network.'''
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = ConvLayer(dtype=dtype, out_dtype=out_dtype)
        self.primary_capsules = PrimaryCaps(dtype=dtype, out_dtype=out_dtype)
        self.digit_capsules = DigitCaps(batch_size, dtype=dtype, out_dtype=out_dtype)
        #self.decoder = Decoder()
                
    def forward(self, images):
        '''Defines the feedforward behavior.
           param images: the original MNIST image input data
           return: output of DigitCaps layer, reconstructed images, class scores
           '''
        batch, n1, n28, n28_ = images.shape
        # assert batch == batch_size
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
    return CapsuleNetwork(batch)