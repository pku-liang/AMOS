_ = None
L = 8


#  (  N,   C,     L,   H,   W,   K,   D,   R,   S, stride_d, stride, padding_d, padding, dilation)
res3d_18_shapes = [
    ( _,   3,     L, 112, 112,  64,   1,   3,   3,        3,      7,         1,       3,        1), # stem

    ( _,  64,     L,  56,  56,  64,   3,   3,   3,        1,      1,         1,       1,        1), # layer1 x 4

    ( _,  64,     L,  56,  56, 128,   1,   1,   1,        2,      2,         0,       0,        1), # layer2 downsample
    
    ( _,  64,     L,  56,  56, 128,   3,   3,   3,        2,      2,         1,       1,        1), # layer2
    ( _, 128,  L//2,  28,  28, 128,   3,   3,   3,        1,      1,         1,       1,        1), # layer2 x 3

    ( _, 128,  L//2,  28,  28, 256,   1,   1,   1,        2,      2,         0,       0,        1), # layer3 downsample
    ( _, 128,  L//2,  28,  28, 256,   3,   3,   3,        2,      2,         1,       1,        1), # layer3
    ( _, 256,  L//4,  14,  14, 256,   3,   3,   3,        1,      1,         1,       1,        1), # layer3 x 3

    ( _, 256,  L//4,  14,  14, 512,   1,   1,   1,        2,      2,         0,       0,        1), # layer4 downsample
    ( _, 256,  L//4,  14,  14, 512,   3,   3,   3,        2,      2,         1,       1,        1), # layer4
    ( _, 256,  L//8,   7,   7, 512,   3,   3,   3,        1,      1,         1,       1,        1), # layer4 x 3
]


if __name__ == "__main__":
    batches = [256]
    beg = 0
    num = 11
    for batch in batches:
        costs = []
        for i, shape in enumerate(res3d_18_shapes[beg:beg+num]):
            (_, C, D, H, W, K, KD, R, S, stride_d, stride, padding_d, padding, dilation) = shape
            N = batch
            print("\n\nProblem size:")
            print("N, C, D, H, W, K, KD, R, S, stride_d, stride, padding_d, padding, dilation")
            print(N, C, D, H, W, K, KD, R, S, stride_d, stride, padding_d, padding, dilation)
            
            total_s = N * D * H * W * K / stride_d / stride / stride
            total_r = KD * C * R * S
            factored_s = total_s / (16 * 4 * 4) / 320
            factored_r = total_r / (16 * 1 * 1)
            cycles = factored_s * factored_r * (16 + 16 + 4)

            costs.append((cycles / 1e9))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)