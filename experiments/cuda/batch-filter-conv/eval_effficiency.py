from math import *


def eval_split(N, C, P, Q, K, R, S, X, Y, Z):
    fred = (N * R * S * ceil(C / Z) * Z * ceil(K / Y) * Y) / (N*R*S*C*K) - 1
    ired = (N * R * (Q + S - 1) * ceil(P/X) * X * ceil(C/Z) * Z) / (N * (P+R - 1) * (Q+S - 1) * C) - 1
    cred = 1 - (N*R*S*P*Q*K*C) / (N * R * S * Q * ceil(P / X) * X * ceil(K / Y) * Y * ceil(C / Z) * Z)
    return fred, ired, cred


def eval_fuse(N, C, P, Q, K, R, S, X, Y, Z):
    fred = (N * ceil(C * R * S / Z) * Z * ceil(K / Y) * Y) / (N * C * R * S * K) - 1
    ired = (N * ceil(P * Q / X) * X * ceil(C * R * S / Z) * Z) / (N * (P + R - 1) * (Q + S - 1) * C) - 1
    cred = 1 - (N * R * S * P * Q * K * C) / (N * ceil(P * Q/ X) * X * ceil(C * R * S / Z) * Z * ceil(K / Y) * Y)
    return fred, ired, cred


arg_list = [
    # batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shape
    # yolo
    (1, 3, 448, 448, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 112, 112, 192, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 192, 56, 56, 128, 192, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1),  # conv4   3
    (1, 256, 56, 56, 256, 256, 1, 1, 1, 1, 0, 1, 1),  # conv5   4
    (1, 256, 56, 56, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv7   6
    (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv8   7
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv9
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv10
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv11
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv12
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv13
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv14
    (1, 512, 28, 28, 512, 512, 1, 1, 1, 1, 0, 1, 1),  # conv15      8
    (1, 512, 28, 28, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv16     9
    (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv17    10
    (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv18     11
    # # (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv19
    # # (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv20
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv21   12
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 2, 1, 1, 1),  # conv22   13
    (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv23     14
    # (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv24
]

wmma_list = [
    (16, 16, 16),
    (32, 8, 16),
    (8, 32, 16)
]

if __name__ == "__main__":
    for batch in [1]:
        print("Batch:", batch)
        for wmma in wmma_list:
            X, Y, Z = wmma
            print("WMMA:", wmma)
            for i, arg in enumerate(arg_list):
                _, C, H, W, K, _, R, S, _, stride, padding, dilation, groups = arg
                N = batch
                kH = (R - 1) * dilation + 1
                kW = (S - 1) * dilation + 1
                P = (H + 2 * padding - kH) // stride + 1
                Q = (W + 2 * padding - kW) // stride + 1 
                sfred, sired, scred = eval_split(N, C, P, Q, K, R, S, X, Y, Z)
                ffred, fired, fcred = eval_fuse(N, C, P, Q, K, R, S, X, Y, Z)
                print("layer %d" % i)
                print("split: image redundancy is %f, filter redundancy is %f, compute redundancy is %f" % (sired, sfred, scred))
                print("fuse: image redundancy is %f, filter redundancy is %f, compute redundancy is %f" % (fired, ffred, fcred))
                print()