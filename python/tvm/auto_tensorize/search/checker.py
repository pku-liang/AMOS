from tvm.contrib import nvcc
from .. import _ffi_api
from ..target import *
from functools import reduce


def get_buffer_size(scope, stmt):
    """
    scope: str
    stmt: Stmt
    """
    return _ffi_api.get_buffer_size(scope, stmt)


def get_thread_extent(stmt):
    """
    stmt: Stmt
    """
    return _ffi_api.get_thread_extent(stmt)


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
        print("Using arch: sm_%d" % arch, flush=True)
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


class MaliCheckScope(object):
    kKernel = 0
    kThreadblock = 1
    kWarp = 2
    kThread = 3


class MaliProgramChecker(Checker):
    def __init__(
            self,
            check_scope=MaliCheckScope.kThreadblock,
            arch="g76"):
        print("Using arch: {}".format(arch), flush=True)
        self.arch_info = Mali(arch=arch)
        self.scope = check_scope
        self.max_shared_mem_bytes_per_block = \
            self.arch_info.get_shared_memory_bytes()
        self.max_register_bytes_per_thread = \
            self.arch_info.get_register_bytes_per_thread()
        self.max_threads_per_block = \
            self.arch_info.max_threads()
        self.warp_size = self.arch_info.get_warp_size()

    def check_shared_memory(self, ir_module):
        for _, v in ir_module.functions.items():
            buffer_map = get_buffer_size("shared", v.body)
            total_size = 0
            for b, size in buffer_map.items():
                total_size += size.value
            if total_size > self.max_shared_mem_bytes_per_block:
                raise CheckError(
                    "Shared memory excess limit bytes: "
                    "{} (required) vs. {} (given)".format(
                        total_size, self.max_shared_mem_bytes_per_block))

    def check_register_per_warp(self, ir_module):
        allow_size = self.max_register_bytes_per_thread * self.warp_size
        for _, v in ir_module.functions.items():
            buffer_map = get_buffer_size("local", v.body)
            total_size = 0
            for b, size in buffer_map.items():
                total_size += size.value
            if total_size > allow_size:
                raise CheckError(
                    "Register excess limit bytes: "
                    "{} (required) vs. {} (given)".format(
                        total_size, allow_size))

    def check_threads_per_block(self, ir_module):
        for _, v in ir_module.functions.items():
            thread_ext = get_thread_extent(v.body)
            total_size = 1
            for t, e in thread_ext.items():
                if str(t).find("threadIdx") == 0:
                    total_size *= e.value
            if total_size > self.max_threads_per_block:
                raise CheckError(
                    "Work group excess limit size: "
                    "{} (required) vs. {} (given)".format(
                        total_size, self.max_threads_per_block))

    def check(self, ir_module):
        if self.scope >= MaliCheckScope.kThreadblock:
            self.check_shared_memory(ir_module)
            self.check_threads_per_block(ir_module)
        if self.scope >= MaliCheckScope.kWarp:
            self.check_register_per_warp(ir_module)
