from pathlib import Path; import sys
sys.path.append(Path(__file__).parent)

try:
  from .test_feature_common import *
except:
  from test_feature_common import *

"""
Bytes: The total number of bytes accessed by this statement.
Unique Bytes: The total number of unique bytes accessed by this statement.
"""

# TODO: test vector add
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

print('dtype:', A.dtype)
assert feature[0]['a']['bytes'] == 512 * 4
assert feature[0]['a']['unique_bytes'] == 512 * 4
assert feature[0]['b']['bytes'] == 512 * 4
assert feature[0]['b']['unique_bytes'] == 512 * 4
assert feature[0]['c']['bytes'] == 512 * 4
assert feature[0]['c']['unique_bytes'] == 512 * 4


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

# feature_ref = build_structured_feature([{
#   'Y': Feature(AccessType.kReadWrite, nelem(Y) * 4, nelem(Y) * 4, 3*220*220),
# }, {
#   'K': Feature(AccessType.kRead, ),
#   'X': Feature(AccessType.kRead, ),
#   'Y': Feature(AccessType.kReadWrite, ),
# },
# ])

# check_features = ['bytes', 'unique_bytes']
# assert structral_equal(feature, feature_ref)

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

# TODO: test more schedules
