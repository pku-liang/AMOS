# Anonymous Code of AMOS @ ASPLOS'22

AMOS is built on-top-of TVM.
# Code structure
```
C++ header files: include/tvm/auto_tensorize/*.h
C++ source files: src/auto_tensorize/*
python files: python/tvm/auto_tensorize/*
test files: tests/python/auto_tensorize/*
tutorial files: tutorials/auto_tensorize/*
```

# Build
install llvm >=v8.0.0
install CUDA toolkit >=v9.0
```sh
download AMOS (denoted as /path/to/amos)
cd /path/to/amos
git submodule init
git submodule update
mkdir build
cd build
cp ../amos_config.cmake .
cmake ..
make -j 8
```

# Environments
python 3.6+
```sh
pip install numpy decorator attrs tornado psutil xgboost cloudpickle
export PYTHONPATH=$PYTHONPATH:/path/to/amos/python
```

# Run (requires Tensor Core GPU)
```sh
cd tutorials/auto_tensorize/auto_tensorize/cuda
python auto_tensorcore_conv2d_fp16.py
```
