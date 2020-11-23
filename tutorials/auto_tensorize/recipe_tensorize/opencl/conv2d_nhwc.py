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
"""
.. conv-mali:
"""

import tvm
from tvm.te import schedule
from tvm import te
from tvm.contrib import nvcc
from tvm import auto_tensorize as at
from tvm import tir
from tempfile import mkstemp
from tvm import rpc
from tvm.contrib import ndk

import os
import numpy as np
import time

from functools import reduce

def test_conv_nhwc(batch=1, in_channel=256, out_channel=512, in_size=14, kernel=3, pad=1, stride=1):
    # Algorithm
    # A = te.placeholder((in_size, in_size, in_channel, batch), name='A', dtype="int8")
    A = te.placeholder((batch, in_size, in_size, in_channel),
                       name='A', dtype="int8")
    # W = te.placeholder((kernel, kernel, in_channel, out_channel), name='W', dtype="int8")
    W = te.placeholder((kernel, kernel, out_channel, in_channel),
                       name='W', dtype="int8")
    out_size = (in_size - kernel + 2 * pad) // stride + 1
    # Pad input
    Apad = te.compute(
        # (in_size + 2*pad, in_size + 2*pad, in_channel, batch),
        (batch, in_size + 2 * pad, in_size + 2 * pad, in_channel),
        # lambda yy, xx, cc, nn: tvm.tir.if_then_else(
        lambda nn, yy, xx, cc: tvm.tir.if_then_else(
            tvm.tir.all(yy >= pad, yy - pad < in_size,
                        xx >= pad, xx - pad < in_size),
            # A[yy - pad, xx - pad, cc, nn], tvm.tir.const(0, "int8")),
            A[nn, yy - pad, xx - pad, cc], tvm.tir.const(0, "int8")),
        name='Apad')
    # Create reduction variables
    rc = te.reduce_axis((0, in_channel), name='rc')
    ry = te.reduce_axis((0, kernel), name='ry')
    rx = te.reduce_axis((0, kernel), name='rx')
    # Compute the convolution
    B = te.compute(
        # (out_size, out_size, out_channel, batch),
        (batch, out_size, out_size, out_channel),
        # lambda yy, xx, ff, nn: te.sum(
        lambda nn, yy, xx, ff: te.sum(
            # Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff],
            Apad[nn, yy * stride + ry, xx * stride + rx, rc] * W[ry, rx, ff, rc],
            axis=[ry, rx, rc]),
        name='B')

    s = te.create_schedule(B.op)
    s[Apad].compute_inline()  # compute Apad inline

    def dump(): return print(tvm.lower(s, [A, W, B]))

    AA = s.cache_read(Apad, 'shared', [B])
    WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(AA, "local", [B])
    WL = s.cache_read(WW, "local", [B])
    BL = s.cache_write(B, "local")

    ################################
    recipe = at.arm_dot_vlen_local_char4()
    compute_key = "" # dummy
    shape_key = ""
    input_names, output_names, nodes, read_graph, feed_graph = \
        at.construct_dag(
            recipe, compute_key, shape_key, [Apad, W], [B])

    # tile consts
    tile = 8
    num_thread = 8
    block_factor = tile * num_thread
    step = 8
    vthread = 2

    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")

    # Split the workloads
    # hi, wi, fi, ni = s[B].op.axis
    ni, hi, wi, fi = s[B].op.axis
    bz = s[B].fuse(hi, wi)
    by, fi = s[B].split(fi, factor=block_factor)
    bx, ni = s[B].split(ni, factor=block_factor)

    # Bind the iteration variables to GPU thread indices
    s[B].bind(bz, block_z)
    s[B].bind(by, block_y)
    s[B].bind(bx, block_x)

    tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
    txz, ni = s[B].split(ni, nparts=vthread)  # virtual thread split
    ty, fi = s[B].split(fi, nparts=num_thread)
    tx, ni = s[B].split(ni, nparts=num_thread)
    # s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)
    s[B].reorder(bz, by, bx, tyz, txz, ty, tx, ni, fi)

    s[B].bind(tyz, thread_yz)
    s[B].bind(txz, thread_xz)
    s[B].bind(ty, thread_y)
    s[B].bind(tx, thread_x)

    # Schedule BL local write
    s[BL].compute_at(s[B], tx)
    # yi, xi, fi, ni = s[BL].op.axis
    ni, yi, xi, fi = s[BL].op.axis
    ry, rx, rc = s[BL].op.reduce_axis
    rco, rci = s[BL].split(rc, factor=step)
    # s[BL].reorder(rco, ry, rx, rci, fi, ni)
    s[BL].reorder(ni, rco, ry, rx, fi, rci)
    # s[BL].reorder(ni, ry, rx, fi, rc)

    # Attach computation to iteration variables
    s[AA].compute_at(s[BL], rx)
    s[WW].compute_at(s[BL], rx)
    # s[AL].compute_at(s[BL], rci)
    # s[WL].compute_at(s[BL], rci)
    s[AL].compute_at(s[BL], fi)
    s[WL].compute_at(s[BL], fi)

    # Schedule for A's shared memory load
    # yi, xi, ci, ni = s[AA].op.axis
    ni, yi, xi, ci = s[AA].op.axis
    ty, ci = s[AA].split(ci, nparts=num_thread)
    tx, ni = s[AA].split(ni, nparts=num_thread)
    # _, ni = s[AA].split(ni, factor=4)
    # s[AA].reorder(ty, tx, yi, xi, ci, ni)
    s[AA].reorder(tx, ni, yi, xi, ty, ci)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    # s[AA].vectorize(ni)  # vectorize memory load

    # Schedule for W's shared memory load
    # yi, xi, ci, fi = s[WW].op.axis
    yi, xi, fi, ci = s[WW].op.axis
    ty, ci = s[WW].split(ci, nparts=num_thread)
    tx, fi = s[WW].split(fi, nparts=num_thread)
    # _, fi = s[WW].split(fi, factor=4)
    # s[WW].reorder(ty, tx, yi, xi, ci, fi)
    s[WW].reorder(ty, tx, yi, xi, fi, ci)
    s[WW].bind(ty, thread_y)
    s[WW].bind(tx, thread_x)
    # s[WW].vectorize(fi)  # vectorize memory load

    arm_dot = recipe.get_intrinsic(reduction_len=step, capsule_key="arm_dot")

    s[BL].tensorize(rci, arm_dot)

    # dump()
    # return

    mod = tvm.lower(s, [A, W, B])
    print(mod)

    target = "opencl"
    target_host = 'llvm -mtriple=aarch64-linux-android'
    device_key = "android"
    rpc_host = "0.0.0.0"
    rpc_port = 9190

    cmds = [
        "adb reverse tcp:9190 tcp:9190",
        "adb forward tcp:5001 tcp:5001",
        "adb shell am start -n org.apache.tvm.tvmrpc/org.apache.tvm.tvmrpc.MainActivity 1> /dev/null 2> /dev/null",
    ]
    os.system("; ".join(cmds))

    print("Connecting...")
    tracker = rpc.connect_tracker(rpc_host, rpc_port)
    remote = tracker.request(device_key, session_timeout=20)
    ctx = remote.context(target)

    print("Allocating...")
    a_np = np.random.randint(0, 10, size=(
        batch, in_size, in_size, in_channel)).astype("int8")
    w_np = np.random.randint(0, 10, size=(
        kernel, kernel, out_channel, in_channel)).astype("int8")
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(
        np.zeros((batch, out_size, out_size, out_channel), dtype=B.dtype), ctx)

    print("Building...")
    func = tvm.build(mod, target=target, target_host=target_host)
    print(func.imported_modules[0].get_source())

    print("Uploading...")
    fd, lib_file = mkstemp(suffix=".so", prefix="gemm")
    os.close(fd)
    func.export_library(lib_file, ndk.create_shared)
    remote.upload(lib_file)
    func = remote.load_module(os.path.split(lib_file)[-1])

    print("Evaluating...")
    func(a, w, b)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    print('Convolution with intrinstic: {} ms'.format(
        evaluator(a, w, b).mean * 1e3))


if __name__ == "__main__":
    test_conv_nhwc()
