# AMOS: Enabling Automatic Mapping for Tensor Computations On Spatial Accelerators with Hardware Abstraction


[**Install**](#install) | [**Tutorials**](#tutorials) | [**Dive into the code**](#dive-into-the-code) | [**Documentations**](#documentations)


## What is AMOS

AMOS is a mapper that can automatically map tensor computations to spatial accelerators via intrinsic.
When we discuss the problem of mapping, we tend to divide the problem into two exclusive parts: hardware-aware mapping and ISA-aware mapping.
Hardware-aware mapping is to map software directly to hardware units at proper spatial-temporal steps.
ISA-aware mapping is to map software to hardware through intermediate instructions called intrinsic.
For example, when we map tensor computations to Tensor Core, we will need to emit Tensor Core instructions (e.g., CUDA WMMA or PTX MMA).
ISA-aware mapping is hard because:
- The users have to configure and transform the compute according to the constraints of intrinsic or hardware.
- It depends on the user to decide how to map software loops to intrinsics. Usually, there are more than one mapping possibility.
But for an inexperienced user, the best choice may be ignored.


## Install
### 1. Download the source code
```sh
cd ~
git clone https://github.com/KnowingNothing/AMOS.git
```
Then get the submodules
```sh
cd AMOS
git submodule update --init --recursive
```

### 2. Prepare the config file
```sh
mkdir build
cd build
cp ../cmake/config.cmake .
```

If you are not familiar with TVM, please stick to the following steps to configure config.cmake, otherwise, just jump to the cmake step. We recommend you to refer to the documents of TVM (https://tvm.apache.org/docs/install/from_source.html) for details.

### 2.1 LLVM settings
Download LLVM source code from https://github.com/llvm/llvm-project to ~/LLVM. You can install LLVM to anywhere you want. Here we choose `~/LLVM/llvm-10`
```sh
mkdir -p ~/LLVM
cd ~/LLVM
git clone git@github.com:llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-10.0.0
mkdir build
cmake -DCMAKE_INSTALL_PREFIX=/home/<your-home-dir>/LLVM/llvm-10 -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld;lldb" ../llvm
make -j 20
make install
```
Then, go back to AMOS directory and modify the config.cmake file.
```sh
cd ~/AMOS/build
vim config.cmake
```
Change the `USE_LLVM` variable to the path to `llvm-config`, i.e., `/home/<your-home-dir>/LLVM/llvm-10/bin/llvm-config` in our example.
### 2.2 CUDA settings
Usually, CUDA toolkit should be installed by the administrater. If you can install CUDA on your own, you can follow the steps of https://developer.nvidia.com/cuda-downloads.
Assume we have CUDA-11.4 installed in `/usr/local/cuda-11.4`.
You can add `/usr/local/cuda-11.4/bin` to your `PATH` so that you have access to `nvcc`.
Then you can further modify the config.cmake file to change `USE_CUDA` variable to value `ON`.
### 2.3 OpenCL settings
To use OpenCL, we can use the OpenCL implementation of Nvidia, which is shipped with CUDA toolkit.
You can simply add `/usr/local/cuda-11.4/lib64` and `/usr/local/cuda-11.4/include` to your `PATH` so that OpenCL libraries can be found.
And modify config.cmake file by changing the value `USE_OPENCL` to `ON`.


### 3. Make and set environments
```sh
cmake ..
make -j 20
```

### 4. Prepare your Python environments
First, we recommend you to use `virualenv` to manage your Python libraries.
If you don't have virtualenv, you can install it locally. If you don't have a pip installed, there are many workarounds, e.g., you can install a Python from source (https://www.python.org/downloads/source/). The details about building Python locally can be found here (https://realpython.com/installing-python/).
```sh
python3 -m pip install --user virtualenv
```
Then use virtualenv to establish your first environment.
```sh
cd ~
mkdir venv
cd venv
python3 -m virtualenv <vir-name> -p python3
```
You can activate your virtual environment by
```sh
source ~/venv/<vir-name>/bin/activate
```
If you find it inconvenient to activate the environment, you can use symbolic link
```sh
mkdir -p .local/bin
cd .local/bin
ln -s /home/<your-home-dir>/venv/<vir-name>/bin/activate <vir-name>
```
Add `/home/<your-home-dir>/.local/bin` to your `PATH` so that you can use a simple `source <vir-name>` to activate your Python environment.
```sh
source <vir-name>
```

Then, install python dependencies of TVM after activating your virtual environment.
```sh
(<vir-name>) pip install numpy decorator attrs tornado psutil xgboost cloudpickle synr
```

At last, setup the environments.
```sh
(<vir-name>) export AMOS_HOME=~/AMOS
(<vir-name>) export PYTHONPATH=$PYTHONPATH:$AMOS_HOME/python
```


## Tutorials
### 1. Conv2d on Tensor Core
This tutorial requires you to use a GPU that supports Tensor Core. GPUs that support Tensor Core should be with Volta/Turing/Ampere architecture.
First of all, import AMOS and tvm.
AMOS is implemented as part of tvm and serves as a function unit of tvm. So we can import AMOS (renamed as auto_tensorize in tvm) from tvm.
```py
import tvm
from tvm import auto_tensorize as at
import numpy as np
```
After that, let's define a Conv2d compute.
```python
def conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    kH = (R - 1) * dilation + 1
    kW = (S - 1) * dilation + 1
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="float16", name="B")

    Pad = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
            A[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype),
        ),
        name="Pad",
    )

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, kH], name="rr")
    rs = tvm.te.reduce_axis([0, kW], name="rs")

    P = (pH - kH) // stride + 1
    Q = (pW - kW) // stride + 1
    Conv = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q: tvm.te.sum(
            (
                Pad[n, rc, p * stride + rr * dilation, q * stride + rs * dilation]
                * B[k, rc, rr, rs]
            ).astype("float32"),
            axis=[rc, rr, rs],
        ),
        name="Conv",
    )
    return [A, B, Conv]

N, H, W, K, C, R, S, stride, padding, dilation = batch, 28, 28, 128, 128, 3, 3, 1, 1, 1
A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
```
The code has no difference from a normal scalar-based conv2d program. Later, AMOS will automatically map this program to Tensor Core. `A`, `B`, `Conv` are the input and output tensors.
We need to trace the program syntax and construct the compute DAG.
```py
target_dag = at.compute_dag_from_tensors([Conv])
```

AMOS needs to perform hardware profiling during mapping exploration and optimization. So we need to set proper measure options.
```py
measure_opt = at.MeasureOptions(target=target, timeout=10, min_repeat_ms=500)
```
The `timeout` is in the unit of second. It is used to control the compilation and execution time. If the compilation or execution overhead exceeds the timeout limit, an error will be reported.
`min_repeat_ms` is used to get accurate performance. Hardware profiling can be inaccurate if we only run the target program a few times. `min_repeat_ms` will force the program to be executed for at least these milliseconds.

Use AMOS to perform mapping exploration:
```py
result = at.auto_tensorize_v4(
        target_dag,
        "cuda",  # code generation target
        "conv2d_tutorial",  # the log file
        measure_opt,
        schedule_log_dir="conv2d_tutorial",
        trials=1200,
        search_group_size=5,
        transform_dump=False,
    )
```
AMOS has multiple interfaces for mapping exploration.
Here we use the latest interface `auto_tensorize_v4`.
The trials we use is 1200 for fast exploration. AMOS will automatically increase the trials to satisfy exploration requirements if the given trials is not enough.
You can also increase the trials to obtain better performance.
When exploring mappings, AMOS will also print a lot of useful message:
```sh
Possible matchings:
0 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:nnn, shape:16x16x16)
1 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:nnn, shape:32x8x16)
2 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:nnn, shape:8x32x16)
3 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:ntn, shape:16x16x16)
4 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:ntn, shape:32x8x16)
5 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:ntn, shape:8x32x16)
6 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:tnn, shape:16x16x16)
7 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:tnn, shape:32x8x16)
8 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:tnn, shape:8x32x16)
9 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:ttn, shape:16x16x16)
10 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:ttn, shape:32x8x16)
11 : MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:ttn, shape:8x32x16)
Logging to devnull...
Totally 35 different mappings for this matching
Logging to devnull...
Totally 35 different mappings for this matching
Catch an infeasible mapping:
{"vmap": [[0, 0, 0, 0, 0, 0, 1], -1]}
Catch an infeasible mapping:
{"vmap": [[0, 0, 0, 0, 1, 0, 0], -1]}
Catch an infeasible mapping:
{"vmap": [[0, 0, 0, 0, 1, 0, 1], -1]}
Catch an infeasible mapping:
{"vmap": [[0, 0, 1, 0, 0, 0, 0], -1]}
Catch an infeasible mapping:
{"vmap": [[0, 0, 1, 0, 0, 0, 1], -1]}
Catch an infeasible mapping:
{"vmap": [[0, 0, 1, 0, 1, 0, 0], -1]}
Catch an infeasible mapping:
{"vmap": [[0, 0, 1, 0, 1, 0, 1], -1]}
Catch an infeasible mapping:
{"vmap": [[0, 1, 0, 0, 0, 0, 0], -1]}
Catch an infeasible mapping:
{"vmap": [[0, 1, 0, 0, 0, 0, 1], -1]}
Catch an infeasible mapping:
{"vmap": [[0, 1, 0, 0, 1, 0, 0], -1]}
Catch an infeasible mapping:
{"vmap": [[0, 1, 1, 0, 0, 0, 0], -1]}
Catch an infeasible mapping:
{"vmap": [[1, 0, 0, 0, 0, 0, 0], -1]}
Catch an infeasible mapping:
{"vmap": [[1, 0, 1, 0, 0, 0, 0], -1]}
Catch an infeasible mapping:
{"vmap": [[1, 1, 0, 0, 0, 0, 0], -1]}
Catch an infeasible mapping:
{"vmap": [[1, 1, 1, 0, 0, 0, 0], -1]}
Total trials: 1200
Num rounds: 10
Num matching: 1
Num mapping: 20
Initial trials per matching: 120
Original weights [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
Original trials for each mapping [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
Current explored matching: MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:nnn, shape:8x32x16)
Its axis mapping:
i: int32 : [n, n, n, p, p, q, q]
j: int32 : [k, k, k, k, k, k, k]
rk: int32 : [rc, rr, rs, rc, rs, rc, rr]
Current explored mapping: {"vmap": [[0, 0, 0, 0, 0, 1, 0], -1]}
Logging to conv2d-fp16-layer-0-batch-1/mapping_(0,0,0,0,0,1,0)_conv2d-fp16-layer-0-batch-1.log...
Loading from file conv2d-fp16-layer-0-batch-1/mapping_(0,0,0,0,0,1,0)_conv2d-fp16-layer-0-batch-1.log...
Load 0 entries! The best known is 10000000000000.000000 ms
Using arch: sm_86
.Y.Y.Y.Y.E
*Y*E*E*E
iteration=1: 6827.986947025252/6827.986947025252
.Y
*E
iteration=2: 1e-10/6827.986947025252
Best record value:6827.986947025252 (larger is better)
Round 1, Match 1, Mapping 1: 6827.986947025252/6827.986947025252(0.1464560503349629 ms), {"vmap": [[0, 0, 0, 0, 0, 1, 0], -1]}, {"inline": 0, "vectorize": 4, "spatial_factors": [[1, 1, 4, 1], [4, 1, 1, 1], [1, 1, 1, 1], [2, 14, 1, 1]], "reduce_factors": [[8, 1, 1], [1, 3, 1], [1, 1, 3]], "last_factors": [[98, 1, 32]], "output_unroll_step": 512, "last_unroll_step": 64}
```
Let's check the message line by line.
First, AMOS tells that there are totally 12 different matches for Tensor Core.
A `match` refers to one applicable intrinsic.
For example, for our float16 Tensor Core, there are 3 different shpaes (16x16x16m 32x8x16m 8x32x16) and 4 layouts (nnn, ntn, ttn, tnn, where `n` means matrix is not transposed and row-major layout is used, and `t` means that matrix is transposed and col-major layout is used).
AMOS chooses one match from the 12 matches by minimizing the number of paddings and redundant computations.
For this tutorial, AMOS chooses `MatchResult(hw_abs_dag:wmma_fp16_fp32, compute:nnn, shape:8x32x16)`.
For this match, there are 35 different mappings.
It may be surprising that there are 35 different methods to map a single Conv2d to Tensor Core.
AMOS can find these mapping (most not discovered before by developers or other compilers) by systematic generation and verification process (details illustrated in [our paper](#cite-us)).
From the 35 mappings, AMOS further rejects 15 infeasible mappings according to the concrete problem size.
For example, the batch size is 1 and it is infeasible to only map batch dimension to Tensor Core, which requires at least 8 elements along matrix row dimension.
After this, AMOS starts evaluating each mapping sequentially.
The profiling results `.Y` means compilation success. `*Y` means execution success.
The performance of the program after mapping is also shown during exploration.

After exploration, we can retrieve the results
```py
schedule_gen = result.sch_gen
schedule_app = result.sch_app

# we store 1/time_cost in file
params, value = result.params, result.perf
cost = at.evaluate_params(schedule_app, params, measure_opt, dump=False)
print("Cost is %f ms" % cost)
```
And check the correctness of results
```py
# retrieve schedule from the record
target_dag = schedule_app.target_dag
inputs = target_dag.get_inputs()
args = inputs + list(target_dag.tensors)
sch = tvm.te.create_schedule([x.op for x in target_dag.tensors])
sch = schedule_app.apply(sch, params)
print(tvm.lower(sch, args, simple_mode=True))
func = tvm.build(sch, args, target)

# test correctness
A, B = inputs
(Conv,) = target_dag.tensors
A_np = np.random.uniform(-10, 10, [int(x) for x in A.shape]).astype(A.dtype)
B_np = np.random.uniform(-10, 10, [int(x) for x in B.shape]).astype(B.dtype)
Conv_np = np.random.uniform(-10, 10, [int(x) for x in Conv.shape]).astype(Conv.dtype)

# use scipy convolve2d api
from tvm.topi.testing import conv2d_nchw_python

Conv_golden = conv2d_nchw_python(
    A_np.astype("float32"), B_np.astype("float32"), stride, padding
)

ctx = tvm.context(target, 0)
A_tvm = tvm.nd.array(A_np, ctx)
B_tvm = tvm.nd.array(B_np, ctx)
Conv_tvm = tvm.nd.array(Conv_np, ctx)
func(A_tvm, B_tvm, Conv_tvm)

from tvm import testing

testing.assert_allclose(Conv_golden, Conv_tvm.asnumpy(), atol=1e-2, rtol=1e-2)
print("Correctness check passed!")
```

We can check the generated code to see if Tensor Core is really used
```py
print(func.imported_modules[0].get_source())
```
The result code is
```c
extern "C" __global__ void default_function_kernel0(half* __restrict__ Pad_vmap_input_cmap_input, half* __restrict__ A) {
  for (int i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner = 0; i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner < 4; ++i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner) {
    Pad_vmap_input_cmap_input[((((((((((((int)blockIdx.x) * 32) + (i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner * 8)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) / 24) * 384) + ((((((int)blockIdx.x) * 4) + i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner) % 3) * 128)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = (((((1 <= ((((((((((((int)blockIdx.x) * 32) + (i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner * 8)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) / 576) * 8) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) / 28) + (((((((((((int)blockIdx.x) * 32) + (i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner * 8)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) % 576) / 24) * 16) + (((int)threadIdx.x) & 15)) % 3))) && (((((((((((((int)blockIdx.x) * 32) + (i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner * 8)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) / 576) * 8) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) / 28) + (((((((((((int)blockIdx.x) * 32) + (i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner * 8)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) % 576) / 24) * 16) + (((int)threadIdx.x) & 15)) % 3)) < 29)) && (1 <= ((((((((((((int)blockIdx.x) * 32) + (i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner * 8)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) / 576) * 8) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) % 28) + (((((int)blockIdx.x) * 4) + i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner) % 3)))) && (((((((((((((int)blockIdx.x) * 32) + (i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner * 8)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) / 576) * 8) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) % 28) + (((((int)blockIdx.x) * 4) + i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner) % 3)) < 29)) ? A[(((((((((((((((((((int)blockIdx.x) * 32) + (i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner * 8)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) % 576) / 24) * 16) + (((int)threadIdx.x) & 15)) / 3) * 784) + ((((((((((((int)blockIdx.x) * 32) + (i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner * 8)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) % 576) / 24) * 16) + (((int)threadIdx.x) & 15)) % 3) * 28)) + ((((((((int)blockIdx.x) * 32) + (i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner * 8)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) / 576) * 8)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + (((((int)blockIdx.x) * 4) + i_o_input_rk_o_input_fused_n_main_input_fused_rs_main_input_fused_i_input_fused_rk_input_fused_outer_outer_inner) % 3)) - 29))] : __float2half_rn(0.000000e+00f));
  }
}

extern "C" __global__ void default_function_kernel3(float* __restrict__ Conv_vmap_output, float* __restrict__ memcpy_dst) {
  for (int n_output_k_output_fused_p_output_fused_q_output_fused_outer_outer_inner = 0; n_output_k_output_fused_p_output_fused_q_output_fused_outer_outer_inner < 4; ++n_output_k_output_fused_p_output_fused_q_output_fused_outer_outer_inner) {
    Conv_vmap_output[(((((((int)blockIdx.x) * 512) + (n_output_k_output_fused_p_output_fused_q_output_fused_outer_outer_inner * 128)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = memcpy_dst[(((((((((((((int)blockIdx.x) * 512) + (n_output_k_output_fused_p_output_fused_q_output_fused_outer_outer_inner * 128)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) % 784) >> 3) * 1024) + ((((((((int)blockIdx.x) * 512) + (n_output_k_output_fused_p_output_fused_q_output_fused_outer_outer_inner * 128)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) / 25088) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + ((((((((int)blockIdx.x) * 512) + (n_output_k_output_fused_p_output_fused_q_output_fused_outer_outer_inner * 128)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) % 25088) / 784)))];
  }
}

extern "C" __global__ void default_function_kernel1(half* __restrict__ B_vmap_input_cmap_input, half* __restrict__ B) {
  for (int j_o_input_rk_o_input_fused_rs_main_input_fused_rk_input_fused_j_input_fused_outer_outer_inner = 0; j_o_input_rk_o_input_fused_rs_main_input_fused_rk_input_fused_j_input_fused_outer_outer_inner < 4; ++j_o_input_rk_o_input_fused_rs_main_input_fused_rk_input_fused_j_input_fused_outer_outer_inner) {
    B_vmap_input_cmap_input[((((((((((((int)blockIdx.x) * 16) + (j_o_input_rk_o_input_fused_rs_main_input_fused_rk_input_fused_j_input_fused_outer_outer_inner * 4)) + ((int)threadIdx.y)) / 48) * 1536) + ((((int)blockIdx.x) % 3) * 512)) + (j_o_input_rk_o_input_fused_rs_main_input_fused_rk_input_fused_j_input_fused_outer_outer_inner * 128)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = B[(((((((((((((int)blockIdx.x) * 16) + (j_o_input_rk_o_input_fused_rs_main_input_fused_rk_input_fused_j_input_fused_outer_outer_inner * 4)) + ((int)threadIdx.y)) / 1152) * 36864) + (((int)threadIdx.x) * 1152)) + ((((((((int)blockIdx.x) * 16) + (j_o_input_rk_o_input_fused_rs_main_input_fused_rk_input_fused_j_input_fused_outer_outer_inner * 4)) + ((int)threadIdx.y)) % 1152) / 48) * 48)) + (j_o_input_rk_o_input_fused_rs_main_input_fused_rk_input_fused_j_input_fused_outer_outer_inner * 12)) + (((int)threadIdx.y) * 3)) + (((int)blockIdx.x) % 3)))];
  }
}

extern "C" __global__ void default_function_kernel2(half* __restrict__ Pad_vmap_input_cmap_input, half* __restrict__ B_vmap_input_cmap_input, float* __restrict__ memcpy_dst) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, float> Conv_vmap_main_cmap_main[1];
  __shared__ half Pad_vmap_input_cmap_input_shared[1536];
  __shared__ half B_vmap_input_cmap_input_shared[6144];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> memcpy_dst1[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::row_major> memcpy_dst2[2];
  (void)nvcuda::wmma::fill_fragment(Conv_vmap_main_cmap_main[0], 0.000000e+00f);
  for (int rk_o_main_outer_outer = 0; rk_o_main_outer_outer < 4; ++rk_o_main_outer_outer) {
    for (int rs_main_main_outer_outer = 0; rs_main_main_outer_outer < 3; ++rs_main_main_outer_outer) {
      __syncthreads();
        ((uint1*)(Pad_vmap_input_cmap_input_shared + (((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) * 128) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)))))[0] = ((uint1*)(Pad_vmap_input_cmap_input + ((((((((((int)blockIdx.x) >> 1) * 18432) + (rk_o_main_outer_outer * 2304)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) * 384)) + (rs_main_main_outer_outer * 128)) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)))))[0];
        ((uint1*)(Pad_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) * 128) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 256))))[0] = ((uint1*)(Pad_vmap_input_cmap_input + (((((((((((int)blockIdx.x) >> 1) * 18432) + (rk_o_main_outer_outer * 2304)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) * 384)) + (rs_main_main_outer_outer * 128)) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 768))))[0];
        ((uint1*)(Pad_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) * 128) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 512))))[0] = ((uint1*)(Pad_vmap_input_cmap_input + (((((((((((int)blockIdx.x) >> 1) * 18432) + (rk_o_main_outer_outer * 2304)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) * 384)) + (rs_main_main_outer_outer * 128)) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1536))))[0];
        ((uint1*)(Pad_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) * 128) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 768))))[0] = ((uint1*)(Pad_vmap_input_cmap_input + (((((((((((int)blockIdx.x) >> 1) * 18432) + (rk_o_main_outer_outer * 2304)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) * 384)) + (rs_main_main_outer_outer * 128)) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 9216))))[0];
        ((uint1*)(Pad_vmap_input_cmap_input_shared + (((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024) / 768) * 768) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) + 2) * 128)) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)))))[0] = ((uint1*)(Pad_vmap_input_cmap_input + (((((((((((int)blockIdx.x) >> 1) * 18432) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024) / 768) * 9216)) + (rk_o_main_outer_outer * 2304)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) + 2) * 384)) + (rs_main_main_outer_outer * 128)) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)))))[0];
        ((uint1*)(Pad_vmap_input_cmap_input_shared + (((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280) / 768) * 768) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) + 4) * 128)) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)))))[0] = ((uint1*)(Pad_vmap_input_cmap_input + (((((((((((int)blockIdx.x) >> 1) * 18432) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280) / 768) * 9216)) + (rk_o_main_outer_outer * 2304)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) >> 7) + 4) * 384)) + (rs_main_main_outer_outer * 128)) + ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) >> 3)) & 7) * 16)) + ((((int)threadIdx.x) & 7) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (rs_main_main_outer_outer * 512)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (rs_main_main_outer_outer * 512)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 256))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (rs_main_main_outer_outer * 512)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 1536))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768) >> 9) * 512) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (rs_main_main_outer_outer * 512)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 3072))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280) >> 9) * 512) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1536))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (rs_main_main_outer_outer * 512)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 4608))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792) >> 9) * 512) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 2048))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (rs_main_main_outer_outer * 512)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 6144))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 2304) >> 9) * 512) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 2304) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 2560))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (rs_main_main_outer_outer * 512)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 7680))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 2816) >> 9) * 512) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 2816) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 3072))))[0] = ((uint1*)(B_vmap_input_cmap_input + ((((((((((int)blockIdx.x) & 1) * 73728) + (rk_o_main_outer_outer * 9216)) + (rs_main_main_outer_outer * 512)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)) + 36864))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 3328) >> 9) * 512) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((((int)blockIdx.x) & 1) * 73728) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 3328) / 3072) * 36864)) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 3584))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((((int)blockIdx.x) & 1) * 73728) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 3584) / 3072) * 36864)) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((int)threadIdx.y) * 64) + ((((int)threadIdx.x) >> 4) * 32))) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 3840) >> 9) * 512) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((((int)blockIdx.x) & 1) * 73728) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 3840) / 3072) * 36864)) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 4096))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((((int)blockIdx.x) & 1) * 73728) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 4096) / 3072) * 36864)) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((int)threadIdx.y) * 64) + ((((int)threadIdx.x) >> 4) * 32))) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 4352) >> 9) * 512) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((((int)blockIdx.x) & 1) * 73728) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 4352) / 3072) * 36864)) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 4608))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((((int)blockIdx.x) & 1) * 73728) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 4608) / 3072) * 36864)) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1536) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((int)threadIdx.y) * 64) + ((((int)threadIdx.x) >> 4) * 32))) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 4864) >> 9) * 512) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((((int)blockIdx.x) & 1) * 73728) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 4864) / 3072) * 36864)) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 5120))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((((int)blockIdx.x) & 1) * 73728) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 5120) / 3072) * 36864)) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 2048) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((int)threadIdx.y) * 64) + ((((int)threadIdx.x) >> 4) * 32))) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 5376) >> 9) * 512) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((((int)blockIdx.x) & 1) * 73728) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 5376) / 3072) * 36864)) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 2304) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 5632))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((((int)blockIdx.x) & 1) * 73728) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 5632) / 3072) * 36864)) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 2560) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((int)threadIdx.y) * 64) + ((((int)threadIdx.x) >> 4) * 32))) + ((((int)threadIdx.x) & 15) * 2)))))[0];
        ((uint1*)(B_vmap_input_cmap_input_shared + ((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 5888) >> 9) * 512) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(B_vmap_input_cmap_input + (((((((((((int)blockIdx.x) & 1) * 73728) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 5888) / 3072) * 36864)) + (rk_o_main_outer_outer * 9216)) + (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 2816) >> 9) * 1536)) + (rs_main_main_outer_outer * 512)) + ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) + 8) * 32)) + ((((int)threadIdx.x) & 15) * 2)))))[0];
      __syncthreads();
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst1[0], ((half *)Pad_vmap_input_cmap_input_shared + (((((int)threadIdx.y) >> 1) * 768))), 16);
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst1[1], ((half *)Pad_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) >> 1) * 768) + 128))), 16);
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst2[0], ((half *)B_vmap_input_cmap_input_shared + (((((int)threadIdx.y) & 1) * 3072))), 32);
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst2[1], ((half *)B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) & 1) * 3072) + 512))), 32);
      (void)nvcuda::wmma::mma_sync(Conv_vmap_main_cmap_main[0], memcpy_dst1[0], memcpy_dst2[0], Conv_vmap_main_cmap_main[0]);
      (void)nvcuda::wmma::mma_sync(Conv_vmap_main_cmap_main[0], memcpy_dst1[1], memcpy_dst2[1], Conv_vmap_main_cmap_main[0]);
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst1[0], ((half *)Pad_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) >> 1) * 768) + 256))), 16);
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst1[1], ((half *)Pad_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) >> 1) * 768) + 384))), 16);
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst2[0], ((half *)B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) & 1) * 3072) + 1024))), 32);
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst2[1], ((half *)B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) & 1) * 3072) + 1536))), 32);
      (void)nvcuda::wmma::mma_sync(Conv_vmap_main_cmap_main[0], memcpy_dst1[0], memcpy_dst2[0], Conv_vmap_main_cmap_main[0]);
      (void)nvcuda::wmma::mma_sync(Conv_vmap_main_cmap_main[0], memcpy_dst1[1], memcpy_dst2[1], Conv_vmap_main_cmap_main[0]);
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst1[0], ((half *)Pad_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) >> 1) * 768) + 512))), 16);
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst1[1], ((half *)Pad_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) >> 1) * 768) + 640))), 16);
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst2[0], ((half *)B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) & 1) * 3072) + 2048))), 32);
      (void)nvcuda::wmma::load_matrix_sync(memcpy_dst2[1], ((half *)B_vmap_input_cmap_input_shared + ((((((int)threadIdx.y) & 1) * 3072) + 2560))), 32);
      (void)nvcuda::wmma::mma_sync(Conv_vmap_main_cmap_main[0], memcpy_dst1[0], memcpy_dst2[0], Conv_vmap_main_cmap_main[0]);
      (void)nvcuda::wmma::mma_sync(Conv_vmap_main_cmap_main[0], memcpy_dst1[1], memcpy_dst2[1], Conv_vmap_main_cmap_main[0]);
    }
  }
  (void)nvcuda::wmma::store_matrix_sync(((float *)memcpy_dst + ((((((((int)blockIdx.x) >> 1) * 2048) + ((((int)threadIdx.y) >> 1) * 1024)) + ((((int)blockIdx.x) & 1) * 512)) + ((((int)threadIdx.y) & 1) * 256)))), Conv_vmap_main_cmap_main[0], 32, nvcuda::wmma::mem_row_major);
}
```
We can see `nvcuda::wmma::mma_sync`, which means that Tensor Core is used.
There are multiple kernels because some kernels are used to do data transform.
In some cases, these kernels can also be fused by AMOS.
The peroformance of this code is `0.031456`ms after 200 tuning. If we wait the full tuning to complete, a better performance can be obtained.



## Dive into the code
The main body of AMOS is put in
```
C++ header files: include/tvm/auto_tensorize/*.h
C++ source files: src/auto_tensorize/*
python files: python/tvm/auto_tensorize/*
tutorial files: tutorials/auto_tensorize/*
```
We also modified some code outside `auto_tensorize` to facilitate the code generation process.
For example
```
src/target/source/codegen_c.h
src/target/source/codegen_c.cc
```
are modified.
The detailed list of modified code outside `auto_tensorize` directory is omitted.


### 1. Hardware abstraction implementation
to be written.
### 2. Mapping generation & exploration part
to be written.

### 3. Mapping exploration
to be written.



## Documentations
to be added.

## Cite us
to be added.

