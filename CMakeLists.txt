cmake_minimum_required(VERSION 3.0.0)
project(TCN-Gemmini)

# options
option(SPIKE "build for spike" ON)
option(BAREMETAL "build baremetal" OFF)
option(MODE_VCU118 "Use VCU118 instead of VC707." OFF)
option(BUILD_TCN_MODEL_ONLY "Build tcn_model only" OFF)
option(DEBUG_BUILD "Enable all printf" ON)
option(PROFILE "Enable profile" OFF)

if (BAREMETAL)
    add_definitions(-DBAREMETAL=1)
    set(CMAKE_C_COMPILER riscv64-unknown-elf-gcc)
    set(CMAKE_CXX_COMPILER riscv64-unknown-elf-g++)
else()
    set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
endif()

if (SPIKE)
    add_definitions(-DSPIKE=1)
endif()

SET(CMAKE_EXE_LINKER_FLAGS "-static")

set(NUM_BLOCKS "1" CACHE STRING "Set # layers")
set(MODE "0" CACHE STRING "TCN Mode. 0: original, 1: DWS, 2: Stateful")
set(HIDDEN_FEATURES "64" CACHE STRING "# hidden features.")

include_directories(
    ${PROJECT_SOURCE_DIR}
)

# source
set(SRC_MODULES
    modules
    modules/tcn_model.cc
    modules/tcnblock.cc
    modules/stateful_conv1d.cc
    modules/layer.cc
    modules/dws_conv1d.cc
    modules/gemm.cc
)

set(SRC_UTIL
    util/util.cc
    util/simple_malloc.cc
)

add_library(
    ${SRC_MODULES}
    ${SRC_UTIL}
)

# tests
if (NOT BUILD_TCN_MODEL_ONLY)
    add_executable(conv1d_test conv1d_test.cc)
    add_executable(stateful_conv1d_test stateful_conv1d_test.cc)
    add_executable(tcnblock_test tcnblock_test.cc)
    add_executable(dws_conv1d_test dws_conv1d_test.cc)

    target_link_libraries(conv1d_test modules)
    target_link_libraries(stateful_conv1d_test modules)
    target_link_libraries(tcnblock_test modules)
    target_link_libraries(dws_conv1d_test modules)
else ()
    message(STATUS "Only builds tcn_model_test")
endif ()

add_executable(tcn_model_test tcn_model_test.cc)

target_link_libraries(tcn_model_test modules)

if (MODE_VCU118)
    message(STATUS "Build for VCU118")
else ()
    message(STATUS "Build for VC707")
endif ()

message(STATUS "# TCN blocks: ${NUM_BLOCKS}")
message(STATUS "# hidden features: ${HIDDEN_FEATURES}")

if (DEBUG_BUILD)
    message(STATUS "Enable printf, assertion")
    add_definitions(-DDEBUG=1)
else ()
    add_definitions(-DNDEBUG=1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
endif ()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MODE_VCU118)
    add_definitions(-DMODE_VCU118=1)
else ()
    add_definitions(-DMODE_VC707=1)
endif()

if (PROFILE)
    add_definitions(-DPROFILE=1)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medany \
    -g \
    -ffast-math \
    -fno-common \
    -fno-builtin-printf \
    -fno-exceptions \
    -march=rv64gc \
    -lm \
    -lgcc \
    -DID_STRING=$(ID_STRING) \
    -DNUM_BLOCKS=${NUM_BLOCKS} \
    -DMODE=${MODE} \
    -DHIDDEN_FEATURES=${HIDDEN_FEATURES}")
