import torch
import numpy as np

def weight_conv(N, I, O, groups, dtype):
  channel_per_group = I // groups
  # out_channel_per_group = O // groups
  A_np = np.random.uniform(-10, 10, [N, I, 1, 1]).astype("float32")
  B_np = np.random.uniform(-10, 10, [O, channel_per_group, 1, 1]).astype("float32")

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

          C_torch = torch.nn.functional.conv2d(A_torch, B_torch, bias=None,
              stride=1, padding=0, groups=groups)

          end.record()
          torch.cuda.synchronize()
          total = start.elapsed_time(end)
          time_record.append(total)
      if i == repeats - 1:
        mean_cost = np.mean(time_record)
  #print("wcv, dtype = %s, A: %s, B: %s, C:%s" % (dtype, A_torch.dtype, B_torch.dtype, C_torch.dtype))
  print(",".join(map(str, [N, I, O, groups, dtype, mean_cost])))



# https://github.com/megvii-model/WeightNet/blob/master/shufflenet_v2.py
# in_channels, out_channels, groups (ksize, stride, padding = 1, 1, 0)
shuffle_v2_cfg = [
    (24, 216, 24),
    (48, 576, 48),
    (56, 504, 56),
    (112, 1008, 112),
    (112, 1344, 112),
    (112, 3136, 112),
    (176, 4928, 176),
    (224, 2016, 224),
    (224, 12544, 224),
    (448, 50176, 448),
]


if __name__ == "__main__":
    assert torch.backends.cudnn.is_available()
    torch.backends.cudnn.enabled = True
    batches = [2**i for i in range(1)]
    beg = 0
    num = len(shuffle_v2_cfg)
    print("batch, in_channels, out_channels, groups, type, cost")
    for dtype in ["FP16", "FP32", "TF32", "FP64", "BF16"]: # "INT8", "BOOL"
      for batch in batches:
          costs = []
          for i, shape in enumerate(shuffle_v2_cfg[beg:beg+num]):
              (I, O, groups) = shape
              N = batch
              weight_conv(N, I, O, groups, dtype)
    print("cudnn: %s" % ("enabled" if torch.backends.cudnn.enabled else "disabled"))
