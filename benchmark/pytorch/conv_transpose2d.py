import torch
import numpy as np
import argparse

def conv_transpose2d(H, W, N, C, K, kernel_size, stride, padding, dtype):
  A_np = np.random.uniform(-10, 10, [N, K, H, W]).astype("float32")
  B_np = np.random.uniform(-10, 10, [K, C, kernel_size, kernel_size]).astype("float32")

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

  global RUN_NUMBER
  number, repeats = RUN_NUMBER

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
        mean_cost = np.mean(time_record)
  print(",".join(map(str, [N, C, H, W, K, kernel_size, kernel_size, stride, padding, dtype, mean_cost])))


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

example_text = """
    example:
        python conv_transpose2d.py --batch 16 --enable_cudnn --number 5 --repeats 5 --begin 0 --num 10 --dtype FP16
        python conv_transpose2d.py --batch 8 --number 10 --repeats 10 --begin 0 --num 5 --dtype TF32
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
    parser.add_argument("--begin", type=int, choices=list(range(len(transpose2d_config))), default=0)
    parser.add_argument(
        "--num",
        type=int,
        choices=list(range(1, len(transpose2d_config) + 1)),
        default=len(transpose2d_config),
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
    RUN_NUMBER = (args.number, args.repeats)
    
    costs = []
    for i, shape in enumerate(transpose2d_config[beg : beg + num]):
        (  N,   C,   H,   W,   K, kernel_size, stride, padding, dilation) = shape
        iH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        iW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        N = batch
        conv_transpose2d(iH, iW, N, C, K, kernel_size, stride, padding, args.dtype)
    print("cudnn: %s" % ("enabled" if torch.backends.cudnn.enabled else "disabled"))