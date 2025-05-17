/**
 * simple_malloc.hh
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
