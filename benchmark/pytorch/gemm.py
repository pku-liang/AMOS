import torch
import numpy as np

def gemm(H, W, K, dtype):
  A_np = np.random.uniform(-10, 10, [H, K]).astype("float32")
  B_np = np.random.uniform(-10, 10, [K, W]).astype("float32")

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

  number = 10
  repeats = 10

  for i in range(repeats):
      time_record = []
      for j in range(number):
          torch.cuda.synchronize()
          start = torch.cuda.Event(enable_timing=True)
          end = torch.cuda.Event(enable_timing=True)
          start.record()

          C_torch = torch.mm(A_torch, B_torch)

          end.record()
          torch.cuda.synchronize()
          total = start.elapsed_time(end)
          time_record.append(total)
      if i == repeats - 1:
        print("Average gemm latency", np.mean(time_record))
        print("Median  gemm latency", np.median(time_record))
  print("GEMM, H = %d, W = %d, K = %d" % (H, W, K))
  print("dtype = %s, A: %s, B: %s, C:%s" % (dtype, A_torch.dtype, B_torch.dtype, C_torch.dtype))
  print("------------------")


if __name__ == "__main__":
  assert torch.backends.cudnn.is_available()
  torch.backends.cudnn.enabled = True
  # H, W, K for [H, K] * [K, W]
  H, W, K = 1024, 1024, 1024
  gemm(H, W, K, "FP16")
  # gemm(H, W, K, "BF16") # Only on A100. On V100, this will throw exception
  gemm(H, W, K, "FP32")
  gemm(H, W, K, "TF32") # Only on A100
  # gemm(H, W, K, "INT8") # No support for INT8 gemm
  # gemm(H, W, K, "BOOL") # No support for BOOL gemm
  gemm(H, W, K, "FP64") # Only on A100

  print("cudnn: %s" % ("enabled" if torch.backends.cudnn.enabled else "disabled"))
