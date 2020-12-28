import torch
import numpy as np

def conv_transpose2d(H, W, N, C, outC, kernel_size, stride, padding, dtype):
  A_np = np.random.uniform(-10, 10, [N, outC, H, W]).astype("float32")
  B_np = np.random.uniform(-10, 10, [outC, C, kernel_size, kernel_size]).astype("float32")

  # What's supported by NVIDIA? Refer to https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html

  # What's supported by pytorch? I don't know
  # Please sudo nvprof them!

  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False

  if dtype == "FP16": # HMMA-16, torch.float16 or torch.half
    A_torch = torch.tensor(A_np).type(torch.float16).cuda()
    B_torch = torch.tensor(B_np).type(torch.float16).cuda()
  elif dtype == "BF16": # HMMA-16, only on NVIDIA A100, torch.bfloat16
    A_torch = torch.tensor(A_np).type(torch.bfloat16).cuda()
    B_torch = torch.tensor(B_np).type(torch.bfloat16).cuda()
  elif dtype == "FP32":
    A_torch = torch.tensor(A_np).type(torch.float32).cuda()
    B_torch = torch.tensor(B_np).type(torch.float32).cuda()
  elif dtype == "TF32": # HMMA-19, NVIDIA A100
    # Please upgrade torch to 1.7; only supported on A100
    # https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    A_torch = torch.tensor(A_np).type(torch.float32).cuda()
    B_torch = torch.tensor(B_np).type(torch.float32).cuda()
  elif dtype == "INT8": # IMMA, but pytorch has no support for INT8 GEMM
    A_torch = torch.tensor(A_np).type(torch.int8).cuda()
    B_torch = torch.tensor(B_np).type(torch.int8).cuda()
  # Pytorch has no int4 type
  elif dtype == "BOOL": # BMMA, but pytorch has no support for GEMM GEMM
    A_torch = torch.tensor(A_np).type(torch.bool).cuda()
    B_torch = torch.tensor(B_np).type(torch.bool).cuda()
  elif dtype == "FP64": # DMMA(FP64), only supported on A100
    A_torch = torch.tensor(A_np).type(torch.float64).cuda()
    B_torch = torch.tensor(B_np).type(torch.float64).cuda()
  else:
    assert False, "wrong type: " + dtype

  number = 10
  repeats = 10

  for i in range(repeats):
      time_record = []
      for j in range(number):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        C_torch = torch.nn.functional.conv_transpose2d(A_torch, B_torch, bias=None, stride=stride, padding=padding)

        end.record()
        torch.cuda.synchronize()
        total = start.elapsed_time(end)
        time_record.append(total)
      if i == repeats - 1:
        print("Average conv_transpose2d latency", np.mean(time_record))
        print("Median  conv_transpose2d latency", np.median(time_record))
  print("conv_transpose2d, dtype = %s, A: %s, B: %s, C:%s" % (dtype, A_torch.dtype, B_torch.dtype, C_torch.dtype))
  print("H, W, C, outC, kernel_size, stride, padding")
  print(",".join(map(str, [H, W, C, outC, kernel_size, stride, padding])))
  print("------------------")


_ = None
#  (  N,   C,   H,   W,   K, kernel_size, stride, padding, dilation)
transpose2d_config = [
    ( _,   3, 112, 112,  64,           3,      7,       3,        1), # stem

    ( _,  64,  56,  56,  64,           3,      1,       1,        1), # layer1 x 4

    ( _,  64,  56,  56, 128,           1,      2,       0,        1), # layer2 downsample
    
    ( _,  64,  56,  56, 128,           3,      2,       1,        1), # layer2
    ( _, 128,  28,  28, 128,           3,      1,       1,        1), # layer2 x 3

    ( _, 128,  28,  28, 256,           1,      2,       0,        1), # layer3 downsample
    ( _, 128,  28,  28, 256,           3,      2,       1,        1), # layer3
    ( _, 256,  14,  14, 256,           3,      1,       1,        1), # layer3 x 3

    ( _, 256,  14,  14, 512,           1,      2,       0,        1), # layer4 downsample
    ( _, 256,  14,  14, 512,           3,      2,       1,        1), # layer4
    ( _, 256,   7,   7, 512,           3,      1,       1,        1), # layer4 x 3
]

if __name__ == "__main__":
    assert torch.backends.cudnn.is_available()
    torch.backends.cudnn.enabled = True
    batches = [2**i for i in range(1)]
    beg = 0
    for batch in batches:
        costs = []
        for i, shape in enumerate(transpose2d_config):
            (  N,   C,   H,   W,   K, kernel_size, stride, padding, dilation) = shape
            iH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            iW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            N = batch
            conv_transpose2d(iH, iW, N, C, K, kernel_size, stride, padding, "FP16")
            conv_transpose2d(iH, iW, N, C, K, kernel_size, stride, padding, "FP32")
            conv_transpose2d(iH, iW, N, C, K, kernel_size, stride, padding, "TF32")
            conv_transpose2d(iH, iW, N, C, K, kernel_size, stride, padding, "FP64")
            # conv_transpose2d(iH, iW, N, C, K, kernel_size, stride, padding, "BF16")
    print("cudnn: %s" % ("enabled" if torch.backends.cudnn.enabled else "disabled"))