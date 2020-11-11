from pathlib import Path
import sys
sys.path.append(Path(__file__).parent)

try:
    from .test_feature_common import *
except:
    from test_feature_common import *


def test_naive_vector_add():
    """
  Test Vector Add
  produce c {
    for (i, 0, 512) {
      c[i] = (a[i] + b[i])
    }
  }
  """

    A, B, C = get_vector_add(512)
    __, feature = get_feature([A, B], [C])
    pprint_dict(feature)

    fA = feature[0]['a']
    fB = feature[0]['b']
    fC = feature[0]['c']

    print('checking access type...')
    assert fA['access_type'] == AccessType.kRead.name
    assert fB['access_type'] == AccessType.kRead.name
    assert fC['access_type'] == AccessType.kWrite.name

    print('checking bytes...')
    assert fA['bytes'] == 512 * 4
    assert fB['bytes'] == 512 * 4
    assert fC['bytes'] == 512 * 4

    print('checking unique bytes...')
    assert fA['unique_bytes'] == nelem(A) * 4
    assert fB['unique_bytes'] == nelem(B) * 4
    assert fC['unique_bytes'] == nelem(C) * 4

    print('checking lines...')
    assert fA['lines'] == 512
    assert fB['lines'] == 512
    assert fC['lines'] == 512

    print('checking unique lines...')
    assert fA['unique_lines'] == nelem(A) * 4 / 128
    assert fB['unique_lines'] == nelem(B) * 4 / 128
    assert fC['unique_lines'] == nelem(C) * 4 / 128

    print('checking reuse type...')
    assert fA['reuse_type'] == ReuseType.kNoReuse.name
    assert fB['reuse_type'] == ReuseType.kNoReuse.name
    assert fC['reuse_type'] == ReuseType.kNoReuse.name

    print('checking reuse counter...')
    assert fA['reuse_counter'] == 1
    assert fB['reuse_counter'] == 1
    assert fC['reuse_counter'] == 1

    print('checking reuse distance...')
    assert fA['reuse_distance'] == 0
    assert fB['reuse_distance'] == 0
    assert fC['reuse_distance'] == 0

    print('checking stride...')
    assert fA['stride'] == 1
    assert fB['stride'] == 1
    assert fC['stride'] == 1

    print('checking topdown...')
    assert fA['topdown'] == 512
    assert fB['topdown'] == 512
    assert fC['topdown'] == 512


def test_naive_conv2d():
    """
  Test Conv2d
  produce Y {
    for (c, 0, 3) {
      for (i, 0, 220) {
        for (j, 0, 220) {
          Y[(((c*48400) + (i*220)) + j)] = 0f
          for (ric, 0, 32) {
            for (rkh, 0, 5) {
              for (rkw, 0, 5) {
                Y[(((c*48400) + (i*220)) + j)] = (Y[(((c*48400) + (i*220)) + j)] + (X[(((((ric*50176) + (i*224)) + (rkh*224)) + j) + rkw)]*K[((((c*800) + (ric*25)) + (rkh*5)) + rkw)]))
              }
            }
          }
        }
      }
    }
  }
  """

    X, K, Y, __ = get_conv2d(3, 32, 224, 224, 5, 5)
    __, feature = get_feature([X, K], [Y])
    pprint_dict(feature)

    print('dtype:', X.dtype)
    print('X.shape:', X.shape)
    print('K.shape:', K.shape)
    print('Y.shape:', Y.shape)

    fY0 = feature[0]['Y']
    fX1 = feature[1]['X']
    fK1 = feature[1]['K']
    fY1 = feature[1]['Y']

    # access type
    print('checking access type...')
    assert fY0['access_type'] == AccessType.kWrite.name
    assert fX1['access_type'] == AccessType.kRead.name
    assert fK1['access_type'] == AccessType.kRead.name
    assert fY1['access_type'] == AccessType.kReadWrite.name

    # bytes
    print('checking bytes...')
    assert fY0['bytes'] == 3 * 220 * 220 * 4
    assert fX1['bytes'] == 3 * 220 * 220 * 32 * 5 * 5 * 4
    assert fK1['bytes'] == 3 * 220 * 220 * 32 * 5 * 5 * 4
    assert fY1['bytes'] == 3 * 220 * 220 * 32 * 5 * 5 * 4 * 2

    # unique bytes
    print('checking unique bytes...')
    assert fY0['unique_bytes'] == nelem(Y) * 4
    assert fX1['unique_bytes'] == nelem(X) * 4
    assert fK1['unique_bytes'] == nelem(K) * 4
    assert fY1['unique_bytes'] == nelem(Y) * 4

    # lines
    print('checking lines...')
    assert fY0['lines'] == 3 * 220 * 220
    assert fX1['lines'] == 3 * 220 * 220 * 32 * 5 * 5
    assert fK1['lines'] == 3 * 220 * 220 * 32 * 5 * 5
    assert fY1['lines'] == 3 * 220 * 220 * 32 * 5 * 5 * 2

    # unique lines
    print('checking unique lines...')
    assert fY0['unique_lines'] == nelem(Y) * 4 // 128
    assert fX1['unique_lines'] == nelem(X) * 4 // 128
    assert fK1['unique_lines'] == nelem(K) * 4 // 128
    assert fY1['unique_lines'] == nelem(Y) * 4 // 128

    # reuse type
    print('checking reuse type...')
    # print(fY0['reuse_type'])  # kNoReuse
    assert fY0['reuse_type'] == ReuseType.kSerialMultipleRead.name
    assert fX1['reuse_type'] == ReuseType.kLoopMultipleRead.name
    assert fK1['reuse_type'] == ReuseType.kLoopMultipleRead.name
    assert fY1['reuse_type'] == ReuseType.kBothReuse.name

    # reuse counter
    print('checking reuse counter...')
    assert fY0['reuse_counter'] == 1
    assert fX1['reuse_counter'] == 3
    assert fK1['reuse_counter'] == 220 * 220
    assert fY1['reuse_counter'] == 32 * 5 * 5 * 2

    # reuse distance
    print('checking reuse distance...')
    assert fY0['reuse_distance'] == 0
    assert fX1['reuse_distance'] == 220 * 220 * 32 * 5 * 5
    assert fK1['reuse_distance'] == 32 * 5 * 5
    assert fY1['reuse_distance'] == 1

    # stride
    print('checking stride...')
    assert fY0['stride'] == 1
    assert fX1['stride'] == 1
    assert fK1['stride'] == 1
    assert fY1['stride'] == 1

    print('checking topdown...')
    assert fY0['topdown'] == 3 * 220 * 220
    assert fX1['topdown'] == 3 * 220 * 220 * 32 * 5 * 5
    assert fK1['topdown'] == 3 * 220 * 220 * 32 * 5 * 5
    assert fY1['topdown'] == 3 * 220 * 220 * 32 * 5 * 5


def test_gpu_naive_conv2d():
  """
  produce Y {
    for (c, 0, 3) {
      // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 220
      // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 220
      Y[(((c*48400) + (blockIdx.x*220)) + threadIdx.x)] = 0f
      for (ric, 0, 32) {
        for (rkh, 0, 5) {
          for (rkw, 0, 5) {
            Y[(((c*48400) + (blockIdx.x*220)) + threadIdx.x)] = (Y[(((c*48400) + (blockIdx.x*220)) + threadIdx.x)] + (X[(((((ric*50176) + (blockIdx.x*224)) + (rkh*224)) + threadIdx.x) + rkw)]*K[((((c*800) + (ric*25)) + (rkh*5)) + rkw)]))
          }
        }
      }
    }
  }
  """
  sch, (X, K, Y) = conv2d_gpu_default(3, 32, 224, 5, 0, 1)
  __, feature = get_feature([X, K], [Y], sch=sch, target='cuda')
  pprint_dict(feature)

  print('dtype:', X.dtype)
  print('X.shape:', X.shape)
  print('K.shape:', K.shape)
  print('Y.shape:', Y.shape)

  fY0 = feature[0]['Y']
  fX1 = feature[1]['X']
  fK1 = feature[1]['K']
  fY1 = feature[1]['Y']

  # access type
  print('checking access type...')
  assert fY0['access_type'] == AccessType.kWrite.name
  assert fX1['access_type'] == AccessType.kRead.name
  assert fK1['access_type'] == AccessType.kRead.name
  assert fY1['access_type'] == AccessType.kReadWrite.name

  # bytes
  print('checking bytes...')
  assert fY0['bytes'] == 3 * 220 * 220 * 4
  assert fX1['bytes'] == 3 * 220 * 220 * 32 * 5 * 5 * 4
  assert fK1['bytes'] == 3 * 220 * 220 * 32 * 5 * 5 * 4
  assert fY1['bytes'] == 3 * 220 * 220 * 32 * 5 * 5 * 4 * 2

  # unique bytes
  print('checking unique bytes...')
  assert fY0['unique_bytes'] == nelem(Y) * 4
  assert fX1['unique_bytes'] == nelem(X) * 4
  assert fK1['unique_bytes'] == nelem(K) * 4
  assert fY1['unique_bytes'] == nelem(Y) * 4

  # lines
  print('checking lines...')
  assert fY0['lines'] == 3 * 220 * 220
  assert fX1['lines'] == 3 * 220 * 220 * 32 * 5 * 5
  assert fK1['lines'] == 3 * 220 * 220 * 32 * 5 * 5
  assert fY1['lines'] == 3 * 220 * 220 * 32 * 5 * 5 * 2

  # unique lines
  print('checking unique lines...')
  assert fY0['unique_lines'] == nelem(Y) * 4 // 128
  assert fX1['unique_lines'] == nelem(X) * 4 // 128
  assert fK1['unique_lines'] == nelem(K) * 4 // 128
  assert fY1['unique_lines'] == nelem(Y) * 4 // 128

  # reuse type
  print('checking reuse type...')
  # print(fY0['reuse_type'])  # kNoReuse
  assert fY0['reuse_type'] == ReuseType.kSerialMultipleRead.name
  assert fX1['reuse_type'] == ReuseType.kLoopMultipleRead.name
  assert fK1['reuse_type'] == ReuseType.kLoopMultipleRead.name
  assert fY1['reuse_type'] == ReuseType.kBothReuse.name

  # reuse counter
  print('checking reuse counter...')
  assert fY0['reuse_counter'] == 1
  assert fX1['reuse_counter'] == 3
  assert fK1['reuse_counter'] == 220 * 220
  assert fY1['reuse_counter'] == 32 * 5 * 5 * 2

  # reuse distance
  print('checking reuse distance...')
  assert fY0['reuse_distance'] == 0
  assert fX1['reuse_distance'] == 220 * 220 * 32 * 5 * 5
  assert fK1['reuse_distance'] == 32 * 5 * 5
  assert fY1['reuse_distance'] == 1

  # stride
  print('checking stride...')
  assert fY0['stride'] == 1
  assert fX1['stride'] == 1
  assert fK1['stride'] == 1
  assert fY1['stride'] == 1

  print('checking topdown...')
  assert fY0['topdown'] == 3
  assert fX1['topdown'] == 3 * 32 * 5 * 5
  assert fK1['topdown'] == 3 * 32 * 5 * 5
  assert fY1['topdown'] == 3 * 32 * 5 * 5
  

def test_gpu_tiled_conv2d():
  """
  produce Y {
    // attr [iter_var(blockIdx.z, , blockIdx.z)] thread_extent = 1
    // attr [Y.local] storage_scope = "local"
    allocate Y.local[float32 * 64]
    // attr [X.shared] storage_scope = "shared"
    allocate X.shared[float32 * 264]
    // attr [K.shared] storage_scope = "shared"
    allocate K.shared[float32 * 9]
    // attr [X.shared.local] storage_scope = "local"
    allocate X.shared.local[float32 * 12]
    // attr [K.shared.local] storage_scope = "local"
    allocate K.shared.local[float32 * 24]
    // attr [iter_var(blockIdx.y, , blockIdx.y)] thread_extent = 55
    // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 4
    // attr [iter_var(threadIdx.z, , threadIdx.z)] thread_extent = 4
    // attr [iter_var(threadIdx.y, , threadIdx.y)] thread_extent = 2
    // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 16
    produce Y.local {
      for (c.c.init, 0, 8) {
        for (i.c.init, 0, 2) {
          for (j.c.init, 0, 4) {
            Y.local[(((c.c.init*8) + (i.c.init*4)) + j.c.init)] = 0f
          }
        }
      }
      for (ric.outer, 0, 32) {
        for (rkh.outer, 0, 5) {
          for (rkw.outer, 0, 2) {
            produce X.shared {
              // attr [iter_var(threadIdx.z, , threadIdx.z)] thread_extent = 4
              // attr [iter_var(threadIdx.y, , threadIdx.y)] thread_extent = 2
              // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 16
              for (ax0.ax1.fused.ax2.fused.inner.inner.inner, 0, 3) {
                if (likely(((floordiv((((threadIdx.y*33) + (threadIdx.x*3)) + ax0.ax1.fused.ax2.fused.inner.inner.inner), 66) + threadIdx.z) < 4))) {
                  if (likely(((floordiv((((threadIdx.y*33) + (threadIdx.x*3)) + ax0.ax1.fused.ax2.fused.inner.inner.inner), 66) + threadIdx.z) < 4))) {
                    if (likely((((((threadIdx.z*66) + (threadIdx.y*33)) + (threadIdx.x*3)) + ax0.ax1.fused.ax2.fused.inner.inner.inner) < 264))) {
                      if (likely(((((threadIdx.y*33) + (threadIdx.x*3)) + ax0.ax1.fused.ax2.fused.inner.inner.inner) < 66))) {
                        if (likely((((threadIdx.x*3) + ax0.ax1.fused.ax2.fused.inner.inner.inner) < 33))) {
                          if (likely(((floordiv((floordiv((((threadIdx.y*33) + (threadIdx.x*3)) + ax0.ax1.fused.ax2.fused.inner.inner.inner), 66) + threadIdx.z), 4) + ric.outer) < 32))) {
                            if (likely(((((blockIdx.x*64) + (rkw.outer*3)) + floormod((((threadIdx.y*33) + (threadIdx.x*3)) + ax0.ax1.fused.ax2.fused.inner.inner.inner), 66)) < 224))) {
                              X.shared[((((threadIdx.z*66) + (threadIdx.y*33)) + (threadIdx.x*3)) + ax0.ax1.fused.ax2.fused.inner.inner.inner)] = X[((((((((floordiv((floordiv((((threadIdx.y*33) + (threadIdx.x*3)) + ax0.ax1.fused.ax2.fused.inner.inner.inner), 66) + threadIdx.z), 4)*50176) + (ric.outer*50176)) + (blockIdx.y*896)) + (rkh.outer*224)) + (floormod((floordiv((((threadIdx.y*33) + (threadIdx.x*3)) + ax0.ax1.fused.ax2.fused.inner.inner.inner), 66) + threadIdx.z), 4)*224)) + (blockIdx.x*64)) + (rkw.outer*3)) + floormod((((threadIdx.y*33) + (threadIdx.x*3)) + ax0.ax1.fused.ax2.fused.inner.inner.inner), 66))]
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            produce K.shared {
              // attr [iter_var(threadIdx.z, , threadIdx.z)] thread_extent = 4
              // attr [iter_var(threadIdx.y, , threadIdx.y)] thread_extent = 2
              // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 16
              if (likely(((floordiv(((threadIdx.y*2) + threadIdx.x), 3) + threadIdx.z) < 3))) {
                if (likely(((floordiv(((threadIdx.y*2) + threadIdx.x), 3) + threadIdx.z) < 3))) {
                  if (likely(((floordiv(((threadIdx.y*2) + threadIdx.x), 3) + threadIdx.z) < 3))) {
                    if (likely(((((threadIdx.z*3) + (threadIdx.y*2)) + threadIdx.x) < 9))) {
                      if (likely((((threadIdx.y*2) + threadIdx.x) < 3))) {
                        if (likely((threadIdx.x < 2))) {
                          if (likely((((rkw.outer*3) + floormod(((threadIdx.y*2) + threadIdx.x), 3)) < 5))) {
                            K.shared[(((threadIdx.z*3) + (threadIdx.y*2)) + threadIdx.x)] = K[((((((floordiv(((threadIdx.y*2) + threadIdx.x), 3)*800) + (threadIdx.z*800)) + (ric.outer*25)) + (rkh.outer*5)) + (rkw.outer*3)) + floormod(((threadIdx.y*2) + threadIdx.x), 3))]
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            produce X.shared.local {
              for (ax1, 0, 2) {
                for (ax2, 0, 6) {
                  if (likely((((((blockIdx.x*64) + (threadIdx.x*4)) + (rkw.outer*3)) + ax2) < 224))) {
                    X.shared.local[((ax1*6) + ax2)] = X.shared[((((threadIdx.y*132) + (ax1*66)) + (threadIdx.x*4)) + ax2)]
                  }
                }
              }
            }
            produce K.shared.local {
              for (ax0, 0, 8) {
                for (ax3, 0, 3) {
                  if (likely((((threadIdx.z*8) + ax0) < 3))) {
                    if (likely((((rkw.outer*3) + ax3) < 5))) {
                      K.shared.local[((ax0*3) + ax3)] = K.shared[(((threadIdx.z*24) + (ax0*3)) + ax3)]
                    }
                  }
                }
              }
            }
            for (rkw.inner.inner, 0, 3) {
              for (c.c, 0, 8) {
                for (i.c, 0, 2) {
                  for (j.c, 0, 4) {
                    if (likely((((rkw.outer*3) + rkw.inner.inner) < 5))) {
                      if (likely((((threadIdx.z*8) + c.c) < 3))) {
                        if (likely(((((blockIdx.x*64) + (threadIdx.x*4)) + j.c) < 220))) {
                          Y.local[(((c.c*8) + (i.c*4)) + j.c)] = (Y.local[(((c.c*8) + (i.c*4)) + j.c)] + (X.shared.local[(((i.c*6) + j.c) + rkw.inner.inner)]*K.shared.local[((c.c*3) + rkw.inner.inner)]))
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    for (c.inner.inner, 0, 8) {
      for (i.inner.inner, 0, 2) {
        for (j.inner.inner, 0, 4) {
          if (likely((((threadIdx.z*8) + c.inner.inner) < 3))) {
            if (likely(((((blockIdx.x*64) + (threadIdx.x*4)) + j.inner.inner) < 220))) {
              Y[((((((((threadIdx.z*387200) + (c.inner.inner*48400)) + (blockIdx.y*880)) + (threadIdx.y*440)) + (i.inner.inner*220)) + (blockIdx.x*64)) + (threadIdx.x*4)) + j.inner.inner)] = Y.local[(((c.inner.inner*8) + (i.inner.inner*4)) + j.inner.inner)]
            }
          }
        }
      }
    }
  }
  """
  sch, (X, K, Y) = conv2d_gpu_tiled(3, 32, 224, 5, 0, 1)
  __, feature = get_feature([X, K], [Y], sch=sch, target='cuda')
  pprint_dict(feature)

  print('dtype:', X.dtype)
  print('X.shape:', X.shape)
  print('K.shape:', K.shape)
  print('Y.shape:', Y.shape)

  # dict_keys(['_stmt_', 'Y.local'])
  # dict_keys(['_stmt_', 'X', 'X.shared'])
  # dict_keys(['_stmt_', 'K', 'K.shared'])
  # dict_keys(['_stmt_', 'X.shared', 'X.shared.local'])
  # dict_keys(['_stmt_', 'K.shared', 'K.shared.local'])
  # dict_keys(['_stmt_', 'K.shared.local', 'X.shared.local', 'Y.local'])
  # dict_keys(['_stmt_', 'Y', 'Y.local'])

  fY0 = feature[0]['Y.local']
  fX1 = feature[1]['X']
  fXs1 = feature[1]['X.shared']
  fK2 = feature[2]['K']
  fKs2 = feature[2]['K.shared']

  # access type
  print('checking access type...')
  assert fY0['access_type'] == AccessType.kWrite.name
  # assert fX1['access_type'] == AccessType.kRead.name
  # assert fK1['access_type'] == AccessType.kRead.name
  # assert fY1['access_type'] == AccessType.kReadWrite.name

  # bytes
  print('checking bytes...')
  assert fY0['bytes'] == 55 * 4 * 4 * 2 * 16 * 8 * 2 * 4 * 4
  # assert fX1['bytes'] == 3 * 220 * 220 * 32 * 5 * 5 * 4
  # assert fK1['bytes'] == 3 * 220 * 220 * 32 * 5 * 5 * 4
  # assert fY1['bytes'] == 3 * 220 * 220 * 32 * 5 * 5 * 4 * 2

  # unique bytes
  print('checking unique bytes...')
  # assert fY0['unique_bytes'] == nelem(Y) * 4 
  # assert fX1['unique_bytes'] == nelem(X) * 4
  # assert fK1['unique_bytes'] == nelem(K) * 4
  # assert fY1['unique_bytes'] == nelem(Y) * 4

  # lines
  print('checking lines...')
  assert fY0['lines'] == 0
  # assert fX1['lines'] == 3 * 220 * 220 * 32 * 5 * 5
  # assert fK1['lines'] == 3 * 220 * 220 * 32 * 5 * 5
  # assert fY1['lines'] == 3 * 220 * 220 * 32 * 5 * 5 * 2

  # unique lines
  print('checking unique lines...')
  assert fY0['unique_lines'] == 0
  # assert fX1['unique_lines'] == nelem(X) * 4 // 128
  # assert fK1['unique_lines'] == nelem(K) * 4 // 128
  # assert fY1['unique_lines'] == nelem(Y) * 4 // 128

  # reuse type
  print('checking reuse type...')
  # ? local storage, consider only 
  assert fY0['reuse_type'] == ReuseType.kBothReuse.name
  # assert fX1['reuse_type'] == ReuseType.kLoopMultipleRead.name
  # assert fK1['reuse_type'] == ReuseType.kLoopMultipleRead.name
  # assert fY1['reuse_type'] == ReuseType.kBothReuse.name

  # reuse counter
  print('checking reuse counter...')
  assert fY0['reuse_counter'] == 55 * 4 * 4 * 2 * 16
  # assert fX1['reuse_counter'] == 3
  # assert fK1['reuse_counter'] == 220 * 220
  # assert fY1['reuse_counter'] == 32 * 5 * 5 * 2

  # reuse distance
  print('checking reuse distance...')
  assert fY0['reuse_distance'] == 8 * 2 * 4
  # assert fX1['reuse_distance'] == 220 * 220 * 32 * 5 * 5
  # assert fK1['reuse_distance'] == 32 * 5 * 5
  # assert fY1['reuse_distance'] == 1

  # stride
  print('checking stride...')
  assert fY0['stride'] == 1
  # assert fX1['stride'] == 1
  # assert fK1['stride'] == 1
  # assert fY1['stride'] == 1

def test_gpu_tiled_conv2d_vthread():
  sch, (X, K, Y) = conv2d_gpu_tiled_vthread(3, 32, 224, 5, 0, 1)
  __, feature = get_feature([X, K], [Y], sch=sch, target='cuda')
  pprint_dict(feature)


def test_gpu_depthwise_cached_block():
  c, n, k, p, s, tc, tw = 32, 64, 3, 1, 1, 16, 4
  sch, (X, K, Y) = depthwise_cached_block(c, n, k, p, s, tc, tw)
  __, feature = get_feature([X, K], [Y], sch=sch, target='cuda')
  pprint_dict(feature)


if __name__ == '__main__':
    print('checking naive vector add...')
    test_naive_vector_add()
    print('checking naive conv2d...')
    test_naive_conv2d()
    print('checking gpu conv2d...')
    test_gpu_naive_conv2d()
    print('checking gpu tiled conv2d...')
    test_gpu_tiled_conv2d()
    # print('checking gpu tiled conv2d vthread...')
    # test_gpu_tiled_conv2d_vthread()
    print('checking gpu depthwise conv2d cached block...')
    test_gpu_depthwise_cached_block()