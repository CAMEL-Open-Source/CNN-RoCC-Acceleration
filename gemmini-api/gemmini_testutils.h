/**
 * gemmini_testutils.h
 * Copyright (C) 2021-2024 CAMELab
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
 
// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

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
#define assert(expr)                                                      \
  if (!(expr)) {                                                          \
    printf("Failed assertion: " #expr "\n  " __FILE__ ":%u\n", __LINE__); \
    exit(1);                                                              \
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

#endif  // SRC_MAIN_C_GEMMINI_TESTUTILS_H
