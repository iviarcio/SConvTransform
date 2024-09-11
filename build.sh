# mkdir build && cd build
cmake -G Ninja .. \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON \
   -DMLIR_DIR=/home/marcio/llvm-project/build/lib/cmake/mlir \
   -DLLVM_DIR=/home/marcio/llvm-project/build/lib/cmake/llvm
cmake --build . --target transform-opt
