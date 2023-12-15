# CNN implementation using RISC-V RoCC interface-based instructions

Target accelerator: gemmini - RISC-V systolic array

`/gemmini-api` consists of files brought from gemmini-rocc-tests.
Few changes for painless compilation
* defined all functions in `gemmini.h` as static functions (may require revision)
* defined elem_t_max, elem_t_min as macro

## Prerequisite
Before using this repo, user should install Chipyard. See guideline at [Chipyard documentation](https://chipyard.readthedocs.io/en/latest/Chipyard-Basics/Initial-Repo-Setup.html#).

Then, since this leverages RISCV's extended instruction set, please follow guideline at [Gemmini's README](https://github.com/ucb-bar/gemmini)

## Build
```
mkdir build
cmake .. [-DOPTION]
make -j N
```

### Available Cmake options
* MODE_VCU118
* MODE_VC707
* BUILD_INTERVAL_ONLY
* DEBUG_BUILD
* PROFILE
* SPIKE
* BAREMETAL
* NUM_BLOCKS
* MODE
* HIDDEN_FEATURES

class Layer : ABC of all layer classes
int prepare() : memory allocations, weight copies
int forward() : forward propagation (inference)
int backward() : backpropagation (training)

## Run
Run executable on RISC-V CPU or Spike.
```
# run on spike
spike --extension=gemmini pk <EXEC>
```
