from ..utils import ERROR

DEFAULT_MAX = 2**12


class HardwareConfig(object):
  def __init__(self):
    self.max_threads = DEFAULT_MAX
    self.max_shared_memory_in_byes = DEFAULT_MAX


HD_CONF_DICT = {}

def register_hardware_config(key, config):
  HD_CONF_DICT[key] = config


def init():
  default_cuda_config = HardwareConfig()
  default_cuda_config.max_threads = 1024
  default_cuda_config.max_shared_memory_in_byes = 48000
  register_hardware_config("default_cuda", default_cuda_config)

  default_llvm_config = HardwareConfig()
  register_hardware_config("default_llvm", default_llvm_config)


def get_hardware_config(hardware_key, target_name):
  if hardware_key in HD_CONF_DICT:
    return HD_CONF_DICT[hardware_key]
  else:
    if "cuda" in target_name:
      return HD_CONF_DICT["default_cuda"]
    elif "llvm" in target_name:
      return HD_CONF_DICT["default_llvm"]
    else:
      ERROR("No supported hardware called %s" % str(hardware_key))


init()