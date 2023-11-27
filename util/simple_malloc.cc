// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include "simple_malloc.hh"

uint32_t SimpleMalloc::counter = 0;
void *SimpleMalloc::malloc(uint32_t) {
  void *ret = (void *)LOCAL_DRAM_START + counter * 0x100'0000;
  ++counter;
  return ret;
}
