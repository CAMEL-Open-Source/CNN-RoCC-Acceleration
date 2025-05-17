/**
 * dws_conv1d.hh
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

#ifndef __DWS_CONV1D_HH__
#define __DWS_CONV1D_HH__

#include <array>

#include "layer.hh"
#include "util/util.hh"

#define DEPTHWISECONV1D_NUM_PARAMS 4
#define DEPTHWISECONV1D_NUM_SCALES 2
class DepthwiseConv1d : public Conv1d {
 private:
  int groups;

 public:
  DepthwiseConv1d(int batch_size, int in_features, int out_features,
                  int in_length, int kernel_size, int stride, int dilation,
                  int padding, int activation = RELU, bool has_bias = true);
  ~DepthwiseConv1d();

  int prepare() override;
  int prepare(std::vector<elem_t *> &, std::vector<int> &) override;
  int seq2col(const elem_t *input) override;
  int forward(elem_t *input, elem_t *output) override;

  void print_weights() override;
};

#define POINTWISECONV1D_NUM_PARAMS 4
#define POINTWISECONV1D_NUM_SCALES 2
class PointwiseConv1d : public Conv1d {
 private:
 public:
  PointwiseConv1d(int batch_size, int in_features, int out_features,
                  int in_length, int activation = RELU, bool has_bias = true);
  ~PointwiseConv1d() {}

  int seq2col(const elem_t *input) override;
};

#define DWSCONV1D_NUM_PARAMS 4
#define DWSCONV1D_NUM_SCALES 2
class DWSConv1d : public Layer {
 private:
  std::array<Conv1d *, 2> convs;
  elem_t *hidden;

  int in_length;
  int out_length;  // length of the output sequence

  int kernel_size;
  int stride;
  int dilation;
  int padding;
  int output_scale;
  int activation;
  bool has_bias;

 public:
  DWSConv1d(int batch_size, int in_features, int out_features, int in_length,
            int kernel_size, int stride, int dilation, int padding,
            int activation = RELU, bool has_bias = true);
  ~DWSConv1d();

  int prepare() override;
  int prepare(std::vector<elem_t *> &, std::vector<int> &) override;

  int forward(elem_t *input, elem_t *output) override;

  void print_weights() override;
};

#endif
