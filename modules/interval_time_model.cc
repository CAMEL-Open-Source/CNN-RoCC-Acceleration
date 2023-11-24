// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include "interval_time_model.hh"

IntervaltimeModel::IntervaltimeModel(int batch_size, int in_feature,
                                     int hidden_feature, int out_category,
                                     int in_length, int num_blocks,
                                     int kernel_size, Mode mode, tiled_matmul_type_t matmul_type) {
  this->batch_size = batch_size;
  this->in_feature = in_feature;
  this->hidden_feature = hidden_feature;
  this->out_category = out_category;
  this->length = in_length;

  this->num_blocks = num_blocks;
  this->kernel_size = kernel_size;
  this->mode = mode;

  std::vector<std::pair<int, int>> list_dilation{{1, 2}, {4, 1}, {2, 4}};

  for (int i = 0; i < num_blocks; i++) {
    int in = i == 0 ? in_feature : hidden_feature;
    if (mode == ORIGINAL) {
      network.emplace_back(new OriginalTCNBlock(
          batch_size, in, hidden_feature, in_length, kernel_size, 1,
          list_dilation.at(i % 3), RELU, true, matmul_type));
    } else if (mode == STATEFUL) {
      network.emplace_back(new StatefulTCNBlock(
          batch_size, in, hidden_feature, in_length, kernel_size, 1,
          list_dilation.at(i % 3), RELU, true, matmul_type));
    } else if (mode == DWS) {
      printf("depthwise separable tcn block not implemented\n");
    } else {
      printf("invalid mode");
    }
  }

  network.emplace_back(new Linear(batch_size, hidden_feature, out_category,
                                  true, NO_ACTIVATION, matmul_type));
}
IntervaltimeModel::~IntervaltimeModel() {
  for (auto layer : network) {
    delete layer;
  }
  for (auto hid : hidden) {
    free(hid);
  }
}

int IntervaltimeModel::prepare() {
  ticks = std::vector<uint64_t>(num_blocks + 1, 0ul);

  // allocate memory for intermediate outputs
  int hidden_size = (mode == STATEFUL) ? batch_size * hidden_feature
                                       : batch_size * length * hidden_feature;
  for (int i = 0; i < num_blocks; i++) {
    hidden.push_back((elem_t *)malloc(sizeof(elem_t) * hidden_size));
  }

  if (mode != STATEFUL) {
    // input for linear
    hidden.push_back(
        (elem_t *)malloc(sizeof(elem_t) * batch_size * hidden_feature));
  }

  for (auto layer : network) {
    layer->prepare();
  }

  return 0;
}

int IntervaltimeModel::prepare(std::vector<elem_t *> &params,
                               std::vector<int> &scales) {
  ticks = std::vector<uint64_t>(num_blocks + 1, 0ul);

  // allocate memory for intermediate outputs
  int hidden_size = (mode == STATEFUL) ? batch_size * hidden_feature
                                       : batch_size * length * hidden_feature;
  for (int i = 0; i < num_blocks; i++) {
    hidden.push_back((elem_t *)malloc(sizeof(elem_t) * hidden_size));
    printf("prepare hidden %d\n", i);
  }

  if (mode != STATEFUL) {
    // input for linear
    hidden.push_back(
        (elem_t *)malloc(sizeof(elem_t) * batch_size * hidden_feature));
  }

  assert(params.size() == num_blocks * TCNBLOCK_NUM_PARAMS + LINEAR_NUM_PARAMS);
  assert(scales.size() == num_blocks * TCNBLOCK_NUM_SCALES + LINEAR_NUM_SCALES);

  auto begin_param = params.begin();
  auto begin_scale = scales.begin();
  for (int i = 0; i < num_blocks; i++) {
    std::vector<elem_t *> subparam(begin_param + i * TCNBLOCK_NUM_PARAMS,
                                   begin_param + (i + 1) * TCNBLOCK_NUM_PARAMS);
    std::vector<int> subscale(begin_scale + i * TCNBLOCK_NUM_SCALES,
                              begin_scale + (i + 1) * TCNBLOCK_NUM_SCALES);

    network[i]->prepare(subparam, subscale);
  }
  std::vector<elem_t *> subparam(begin_param + num_blocks * TCNBLOCK_NUM_PARAMS,
                                 params.end());
  std::vector<int> subscale(begin_scale + num_blocks * TCNBLOCK_NUM_SCALES,
                            scales.end());
  network[num_blocks]->prepare(subparam, subscale);

  return 0;
}

int IntervaltimeModel::forward(elem_t *input, elem_t *output) {
  elem_t *layer_in;

  layer_in = input;
  for (int i = 0; i < num_blocks; i++) {
#if DEBUG
    printf("tcnblock %d\n", i);
#endif
    network[i]->forward(layer_in, hidden[i]);
    ticks.at(i) = read_cycles();
    layer_in = hidden[i];
  }

  if (mode != STATEFUL) {
    // select the last time index
    for (int b = 0; b < batch_size; b++) {
      for (int h = 0; h < hidden_feature; h++) {
        hidden[num_blocks][ARRAY_INDEX_2D(hidden_feature, b, h)] =
            layer_in[ARRAY_INDEX_3D(length, hidden_feature, b, length - 1, h)];
      }
    }

    layer_in = hidden[num_blocks];
  }
#if DEBUG
  printf("linear\n");
#endif
  network[num_blocks]->forward(layer_in, output);
  ticks.at(num_blocks) = read_cycles();

  return 0;
}

void IntervaltimeModel::print_weights() {
  for (auto layer : network) {
    layer->print_weights();
  }
}
