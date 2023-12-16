# CNN implementation using RISC-V RoCC interface-based instructions

Target accelerator: gemmini - RISC-V systolic array

## Getting started
Please refer [docs/getting-started.md](docs/getting-started.md).


## Important class & functions
* `class Layer` : ABC of all layer classes
* `int prepare()` : memory allocations, weight copies
* `int forward()` : forward propagation (inference)
* `int backward()` : backpropagation (training)

## Note
`/gemmini-api` consists of files brought from gemmini-rocc-tests.
Few changes for painless compilation
* defined all functions in `gemmini.h` as static functions (may require revision)
* defined elem_t_max, elem_t_min as macro

## Ongoing
We are extending this project to cover hwacha (RISCV-based vector processing accelerator) as well as gemmini.

## Acknowledgment
This project is funded by IITP 2021-0-00524
