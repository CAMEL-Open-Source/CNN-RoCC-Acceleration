// See LICENSE for license details.

#ifndef __GEMMINI_API_GEMMINI_TESTUTILS_H__
#define __GEMMINI_API_GEMMINI_TESTUTILS_H__

#undef abs

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "gemmini.h"
#include "gemmini_params.h"

#ifdef BAREMETAL
#undef assert
#define assert(expr)                                                           \
  if (!(expr)) {                                                               \
    printf("Failed assertion: " #expr "\n  " __FILE__ ":%u\n", __LINE__);      \
    exit(1);                                                                   \
  }
#endif

// #define GEMMINI_ASSERTIONS

inline int rand() {
  static uint32_t x = 777;
  x = x * 1664525 + 1013904223;
  return x >> 24;
}

#ifdef ELEM_T_IS_FLOAT
inline double rand_double() {
  double a = (double)(rand() % 128) / (double)(1 + (rand() % 64));
  double b = (double)(rand() % 128) / (double)(1 + (rand() % 64));
  return a - b;
}
#endif

inline uint64_t read_cycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;

  // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbff8);
  // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbffc);
  // return *mtime;
}

#undef abs

#endif // SRC_MAIN_C_GEMMINI_TESTUTILS_H
