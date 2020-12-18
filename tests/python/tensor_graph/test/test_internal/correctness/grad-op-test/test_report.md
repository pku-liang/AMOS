## Tested Ops

| op name | case No. | grad to | configs | gradient | build | correctness |
| --- | --- | --- | --- | --- | --- | --- |
| GEMM | 1 | A | | yes | yes | rtol=1e-5, atol<1e-30 |
| Conv2d | 1 | A | st=1, pad=0, group=1, dilation=1 | yes | yes | rtol=1e-3, atol<1e-30|
| Conv2d in topi | 1 | A | st=1, pad=0, group=1, dilation=1 | yes | yes | rtol=1e-5, atol=1e-4 |
| Conv2d | 2 | A | st=2, pad=0, group=1, dilation=1 | yes | yes | rtol=1e-3, atol<1e-30 |
| Conv2d | 3 | A | st=2, pad=0, group=2, dilation=1 | yes | yes | rtol=1e-3, atol<1e-30|
| Flatten | 1 | A | | yes | yes | rtol<1e-30, atol<1e-30 |
| Flatten | 2 | A | use intermediate | yes | yes | rtol<1e-30, atol<1e-30 |
| Downcast | 1 | A | | yes | yes | rtol<1e-30, atol<1e-30 |
| Broadcast | 1 | A | | yes | yes | rtol=1e-6, atol<1e-30 |
| !!!Broadcast | 2 | A | use 0 as index | no | no | NAN |
| Padding | 1 | A | | yes | yes | rtol<1e-30, atol<1e-30 |
| AvgPool | 1 | A | | yes | yes | rtol<1e-30, atol<1e-30|
| Softmax | 1 | A | | yes | yes | rtol=1e-5, atol=1e-6 |
| Maxpool | 1 | A | | yes | yes | rtol=1e-30, atol=1e-5 |
| Tanh | 1 | A | | yes | yes | rtol=1e-7, atol=1e-6|
| ReLU | 1 | A | | yes | yes | rtol<1e-30, atol<1e-30 |
| Mse_loss | 1 | A | | yes | yes | rtol<1e-30, atol<1e-30 |
| Cross_entropy | 1 | A | | yes | yes | rtol<1e-30, atol=1e-9 |
| Concatenate | 1 | A,B | | yes | yes | rtol<1e-30, atol<1e-30 |
| !!!Concatenate | 2 | A,B | if-then-else | no | no | NAN |
| !!!Concatenate | 3 | A,B | if-then-else | no | no | NAN |
| Squeeze | 1 | A | [H1, 1, 1, H2] -> [H1, H2] | yes | yes | rtol<1e-30, atol<1e-30 |
| !!!Reshape | 1 | A | 3-dim, two % in compute (workaround exists) | yes | NO | NAN |
| sub_expr | 1 | A | A[h, w] * 4 - A[h, w] * A[h, w] | yes | yes | rtol<1e-30, atol<1e-30 |
| !!!sub_expr | 2 | A | replace A[h, w] with A[h, w, 0] | NO | NO | NAN |