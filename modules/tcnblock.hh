// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#ifndef __TCNBLOCK_HH__
#define __TCNBLOCK_HH__

#include "dws_conv1d.hh"
#include "layer.hh"
#include "stateful_conv1d.hh"

enum Mode { ORIGINAL, DWS, STATEFUL };

#define TCNBLOCK_NUM_PARAMS 6
#define TCNBLOCK_NUM_SCALES 4
class BaseTCNBlock : public Layer {
 protected:
  Layer *conv1;
  Layer *conv2;
  Conv1d *conv1x1;
  int length;
  int res_shift;

  int hidden_size;

  static constexpr int CONV1_PARAM_START_IDX = 0;
  static constexpr int CONV2_PARAM_START_IDX = 2;
  static constexpr int CONV1X1_PARAM_START_IDX = 4;
  static constexpr int CONV1_SCALE_START_IDX = 1;
  static constexpr int CONV2_SCALE_START_IDX = 2;
  static constexpr int CONV1X1_SCALE_START_IDX = 3;

  elem_t *conv1_out;
  elem_t *conv2_out;
  elem_t *conv1x1_out;

 public:
  BaseTCNBlock(int batch_size, int in_features, int out_features,
               int in_length);
  ~BaseTCNBlock();
  BaseTCNBlock(BaseTCNBlock &&);
  int prepare() override;
  int prepare(std::vector<elem_t *> &, std::vector<int> &) override;
  int forward(elem_t *input, elem_t *output) override;

  void print_weights() override;
};

class OriginalTCNBlock : public BaseTCNBlock {
 public:
  OriginalTCNBlock(int batch_size, int in_features, int out_features,
                   int in_length, int kernel_size, int stride,
                   std::pair<int, int> dilation, int activation,
                   bool has_bias = true, tiled_matmul_type_t matmul_type = WS);
};

class StatefulTCNBlock : public BaseTCNBlock {
 public:
  StatefulTCNBlock(int batch_size, int in_features, int out_features,
                   int in_length, int kernel_size, int stride,
                   std::pair<int, int> dilation, int activation,
                   bool has_bias = true, tiled_matmul_type_t matmul_type = WS);
  int forward(elem_t *input, elem_t *output) override;
};

#endif
