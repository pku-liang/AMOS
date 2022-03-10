import torch
import numpy as np
import argparse

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

  global RUN_NUMBER
  number, repeats = RUN_NUMBER

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

example_text = """
    example:
        python wcv.py --batch 1024 --enable_cudnn --number 5 --repeats 5 --begin 0 --num 11 --dtype FP16
        python wcv.py --batch 512 --number 10 --repeats 10 --begin 0 --num 5 --dtype TF32
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
    parser.add_argument("--begin", type=int, choices=list(range(len(shuffle_v2_cfg))), default=0)
    parser.add_argument(
        "--num",
        type=int,
        choices=list(range(1, len(shuffle_v2_cfg) + 1)),
        default=len(shuffle_v2_cfg),
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

    print("batch, in_channels, out_channels, groups, type, cost")
    costs = []
    for i, shape in enumerate(shuffle_v2_cfg[beg:beg+num]):
        (I, O, groups) = shape
        N = batch
        weight_conv(N, I, O, groups, args.dtype)
    print("cudnn: %s" % ("enabled" if torch.backends.cudnn.enabled else "disabled"))
