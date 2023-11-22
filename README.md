# CNN implementation using RISC-V RoCC interface-based instructions

Target accelerator: gemmini - RISC-V systolic array

`/gemmini-api` consists of files brought from gemmini-rocc-tests.
Few changes for painless compilation
* defined all functions in `gemmini.h` as static functions (may require revision)
* defined elem_t_max, elem_t_min as macro

## build
```
mkdir build
cmake .. [-DOPTION]
# Available cmake options :
# MODE_VCU118, MODE_VC707, BUILD_INTERVAL_ONLY, DEBUG_BUILD, PROFILE,
# NUM_BLOCKS, MODE, HIDDEN_FEATURES
make -j N
```

class Layer : ABC of all layer classes
int prepare() : memory allocations, weight copies
int forward() : forward propagation (inference)
int backward() : backpropagation (training)
