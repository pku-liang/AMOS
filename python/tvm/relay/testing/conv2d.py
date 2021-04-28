# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
from tvm import relay
from .init import create_workload
from . import layers as wrapper


def get_net(
    N, C, H, W, K, R, S,
    stride, padding, dilation,
    groups,
    dtype="float32"
):
    """
    Parameters
    ----------
    """
    data = relay.var("data", shape=[N, C, H, W], dtype=dtype)
    output = wrapper.conv2d(
                data=data,
                kernel_size=(R, S),
                strides=(stride, stride),
                padding=(padding, padding),
                channels=K,
                dilation=(dilation, dilation),
                groups=groups,
                name="conv",
            )
    args = relay.analysis.free_vars(output)
    return relay.Function(args, output)


def get_workload(
    N, C, H, W, K, R, S,
    stride, padding, dilation,
    groups,
    dtype="float32"
):
    """Get benchmark workload for VGG nets.

    Parameters
    ----------
    """
    net = get_net(N, C, H, W, K, R, S,
                  stride, padding, dilation,
                  groups,  dtype)
    return create_workload(net)
