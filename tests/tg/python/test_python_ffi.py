from pathlib import Path; import sys
sys.path.append(Path(__file__).parent)

from .test_feature_common import *

A, B, C = get_vector_add(512)

features, structured_features = get_feature([A, B], [C])
print(f"Flattened features: {features}")
print(f"Structured features: ")
pprint_dict(structured_features)
