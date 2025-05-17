/**
 * tcn_model.hh
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

#ifndef __TCN_MODEL_HH__
#define __TCN_MODEL_HH__

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include <cstring>
#include <string>
#include <vector>

#include "gemmini-api/gemmini_testutils.h"
#include "layer.hh"
#include "tcnblock.hh"
#include "util/util.hh"

class IntervaltimeModel {
 private:
  int batch_size;
  int in_feature;
  int hidden_feature;
  int out_category;
  int length;

  int num_blocks;
  int kernel_size;
  Mode mode;

  std::vector<Layer *> network;  // layers
  std::vector<elem_t *> hidden;  // intermediate outputs

  std::vector<uint64_t> ticks;

 public:
  IntervaltimeModel(int, int, int, int, int, int, int, Mode = ORIGINAL,
                    tiled_matmul_type_t = WS);
  ~IntervaltimeModel();

  int prepare();
  int prepare(std::vector<elem_t *> &, std::vector<int> &);
  int forward(elem_t *, elem_t *);

  std::vector<uint64_t> &getTicks() { return ticks; }

  void print_weights();
};

#endif
