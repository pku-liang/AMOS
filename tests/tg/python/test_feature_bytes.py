from pathlib import Path; import sys
sys.path.append(Path(__file__).parent)

from .test_feature_common import *

"""
Bytes: The total number of bytes accessed by this statement.
Unique Bytes: The total number of unique bytes accessed by this statement.
"""

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

# assert feature[0]['Y']['bytes'] == nelem(Y) * 4
# assert feature[0]['Y']['unique_bytes'] == nelem(Y) * 4
# assert feature[1]['K']['bytes'] == 3 * 220 * 220 * 32 * 5 * 5 * 4
# assert feature[1]['K']['unique_bytes'] == nelem(K) * 4
# assert feature[1]['X']['bytes'] == 224 * 224 * 3 * 4
# assert feature[1]['X']['unique_bytes'] == nelem(X) * 4
# assert feature[1]['Y']['bytes'] == 512 * 4
# assert feature[1]['Y']['unique_bytes'] == nelem(Y) * 4

feature_ref = build_structured_feature([{
  'Y': Feature(),
}, {
  'K': Feature(),
  'X': Feature(),
  'Y': Feature(),
},
])

check_features = ['bytes', 'unique_bytes']
assert structral_equal(feature, feature_ref)