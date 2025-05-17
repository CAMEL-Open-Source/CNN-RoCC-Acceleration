/**
 * stateful_conv1d.hh
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

#ifndef __STATEFUL_CONV1D_HH__
#define __STATEFUL_CONV1D_HH__

#include "gemmini-api/gemmini_testutils.h"
#include "layer.hh"

#define STATEFUL_CONV1D_NUM_PARAMS 2
#define STATEFUL_CONV1D_NUM_SCALES 1
class StatefulConv1d : public Conv1d {
  /* bit different from typical conv1d since this applies padding only on left
   * side rather than both side*/
 private:
  elem_t *old_in;     // [batch_size][kernel_reach-1][in_features]
  elem_t *in_merged;  // [batch_size][kernel_reach][in_features]

 public:
  StatefulConv1d(int batch_size, int in_features, int out_features,
                 int in_length, int kernel_size, int stride, int dilation,
                 int padding, int activation = RELU, bool has_bias = true,
                 tiled_matmul_type_t matmul_type = WS);
  ~StatefulConv1d();

  int prepare() override;
  int prepare(std::vector<elem_t *> &, std::vector<int> &) override;
  int forward(elem_t *input,
              elem_t *output) override;  // input : [batch_size][in_features]
  int seq2col(const elem_t *input) override;
  int merge_and_save(elem_t *);
  int warmup(elem_t *input, elem_t *output);
};

#endif
