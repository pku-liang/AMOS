from .. import _ffi_api
from ..target import *


def get_buffer_size(scope, stmt):
    """
    scope: str
    stmt: Stmt
    """
    return _ffi_api.get_buffer_size(scope, stmt)


class CheckError(Exception):
    pass


class Checker(object):
    def check(self, ir_module):
        raise NotImplementedError()


class CUDACheckScope(object):
    kKernel = 0
    kThreadblock = 1
    kWarp = 2
    kThread = 3


class CUDAProgramChecker(Checker):
    def __init__(
        self,
        check_scope=CUDACheckScope.kThread,
        arch=70):
        self.arch_info = CUDA(arch=arch)
        self.scope = check_scope
        self.max_shared_mem_bytes_per_block = self.arch_info.get_shared_memory_bytes()
        self.max_register_bytes_per_thread = self.arch_info.get_register_bytes_per_thread()
        self.warp_size = self.arch_info.get_warp_size()

    def check_shared_memory(self, ir_module):
        for k, v in ir_module.functions.items():
            buffer_map = get_buffer_size("shared", v.body)
            total_size = 0
            for b, size in buffer_map.items():
                total_size += size.value
            if total_size > self.max_shared_mem_bytes_per_block:
                raise CheckError("Shared memory excess limit bytes: %d (required) vs. %d (given)" % (
                    total_size, self.max_shared_mem_bytes_per_block))

    def check_register_per_warp(self, ir_module):
        allow_size = self.max_register_bytes_per_thread * self.warp_size
        for k, v in ir_module.functions.items():
            buffer_map = get_buffer_size("local", v.body)
            total_size = 0
            for b, size in buffer_map.items():
                total_size += size.value
            if total_size > allow_size:
                raise CheckError("Register excess limit bytes: %d (required) vs. %d (given)" % (
                    total_size, allow_size))

    def check(self, ir_module):
        if self.scope >= CUDACheckScope.kThreadblock:
            self.check_shared_memory(ir_module)
        if self.scope >= CUDACheckScope.kWarp:
            self.check_register_per_warp(ir_module)