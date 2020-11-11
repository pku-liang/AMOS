import tvm
import tvm.te as te
import tvm.tg as tg

import numpy as np

def pprint_dict(d):
  import json
  print(json.dumps(d, indent=2, sort_keys=False))
