// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include <stdint.h>

#define LOCAL_DRAM_START 0x4'4000'0000
#define LOCAL_DRAM_END 0x8'0000'0000

class SimpleMalloc {
 public:
  static uint32_t counter;
  static void *malloc(uint32_t);
};

static void *malloc(uint32_t len) {
  SimpleMalloc s;

  return s.malloc(len);
}
