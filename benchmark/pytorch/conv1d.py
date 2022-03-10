import torch
import numpy as np
import argparse

def conv1d(N, C, L, K, KL, stride, padding, dilation, dtype, number, repeats):
  A_np = np.random.uniform(-10, 10, [N, C, L]).astype("float32")
  B_np = np.random.uniform(-10, 10, [K, C, KL]).astype("float32")

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
  
  for i in range(repeats):
      time_record = []
      for j in range(number):
          torch.cuda.synchronize()
          start = torch.cuda.Event(enable_timing=True)
          end = torch.cuda.Event(enable_timing=True)
          start.record()

          C_torch = torch.nn.functional.conv1d(A_torch, B_torch, bias=None, stride=stride, 
            padding=padding, dilation=dilation)

          end.record()
          torch.cuda.synchronize()
          total = start.elapsed_time(end)
          time_record.append(total)
      if i == repeats - 1:
        mean_cost = np.mean(time_record)
  # print("conv1d, dtype = %s, A: %s, B: %s, C:%s" % (dtype, A_torch.dtype, B_torch.dtype, C_torch.dtype))
  print(",".join(map(str, [N, C, L, K, KL, stride, padding, dilation, dtype, mean_cost])))


byte_net_shapes = [
  # (   C,   L,   K,  KL, stride,padding, dilation)
    ( 512, 892, 512,   3,       1,     2,        1),
    ( 512, 892,1024,   1,       1,     0,        1),
    (1024, 892, 512,   1,       1,     0,        1),
    ( 512, 892, 512,   3,       1,     4,        2),
    ( 512, 892, 512,   3,       1,     8,        4),
    ( 512, 892, 512,   3,       1,    16,        8),
    ( 512, 892, 512,   3,       1,    32,       16),
    (1024, 892, 250,   1,       1,     0,        1)
]

example_text = """
    example:
        python conv1d.py --batch 16 --enable_cudnn --number 5 --repeats 5 --begin 0 --num 8 --dtype FP16
        python conv1d.py --batch 8 --number 10 --repeats 10 --begin 0 --num 4 --dtype TF32
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--begin", type=int, choices=list(range(len(byte_net_shapes))), default=0)
    parser.add_argument(
        "--num",
        type=int,
        choices=list(range(1, len(byte_net_shapes) + 1)),
        default=len(byte_net_shapes),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["FP16", "FP32", "TF32", "FP64", "BF16", "INT8", "BOOL"],
        default="FP16",
    )

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False

    batch = args.batch
    beg = args.begin
    num = args.num

    costs = []
    for i, shape in enumerate(byte_net_shapes[beg:beg+num]):
        (C, L, K, KL, stride, padding, dilation) = shape
        N = batch
        conv1d(N, C, L, K, KL, stride, padding, dilation, args.dtype, args.number, args.repeats)
    print("cudnn: %s" % ("enabled" if torch.backends.cudnn.enabled else "disabled"))
