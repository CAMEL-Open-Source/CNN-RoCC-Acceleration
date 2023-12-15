// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include "dws_conv1d.hh"

#include "gemmini-api/gemmini_testutils.h"

PointwiseConv1d::PointwiseConv1d(int batch_size, int in_features,
                                 int out_features, int in_length,
                                 int activation, bool has_bias)
    : Conv1d(batch_size, in_features, out_features, in_length,

             1, 1, 1, 0, activation, has_bias) {}

int PointwiseConv1d::seq2col(const elem_t *input) {
  // input is (in_feature, batch_size, length)
  // we may want to change the order of loop

  int patch_row = 0;
  // iterate over batch
  for (int n_batch = 0; n_batch < batch_size; n_batch++) {
    // iterate over image coordinates
    for (int im_col = -padding; im_col < in_length; im_col++) {
      // iterate over filter coordinates
      int patch_col = 0;
      int filter_col = 0;
      for (int im_channel = 0; im_channel < in_features; im_channel++) {
        int pixel_col = im_col;

        if (pixel_col < 0 || pixel_col >= in_length) {
          // output[patch_row][patch_col] = 0;
        }
        else {
          seq2col_out[ARRAY_INDEX_2D(patch_size, patch_row, patch_col)] =
              input[ARRAY_INDEX_3D(batch_size, in_length, im_channel, n_batch,
                                   pixel_col)];
        }

        patch_col++;
      }

      patch_row++;
    }
  }
  return 0;
}

DepthwiseConv1d::DepthwiseConv1d(int batch_size, int in_features,
                                 int out_features, int in_length,
                                 int kernel_size, int stride, int dilation,
                                 int padding, int activation, bool has_bias)
    : Conv1d(batch_size, in_features, out_features, in_length, kernel_size,
             stride, dilation, padding, activation, has_bias) {
  groups = in_features;  // only for mobilenet

  // I didn't test other scenario and we don't need it
  assert(in_features == out_features);

  n_patches *= groups;
  patch_size /= groups;
}

DepthwiseConv1d::~DepthwiseConv1d() {}

int DepthwiseConv1d::prepare() {
  seq2col_out = (elem_t *)malloc(sizeof(elem_t) * n_patches * patch_size);
  memset(seq2col_out, 0, sizeof(elem_t) * n_patches * patch_size);

  weight = (elem_t *)malloc(sizeof(elem_t) * groups * patch_size * 1);
  bias = (elem_t *)malloc(sizeof(elem_t) * out_features);
  memset(weight, 0, sizeof(elem_t) * groups * patch_size * 1);
  memset(bias, 0, sizeof(elem_t) * out_features);

  output_scale = 1;

  return 0;
}

int DepthwiseConv1d::prepare(std::vector<elem_t *> &params,
                             std::vector<int> &scales) {
  seq2col_out = (elem_t *)malloc(sizeof(elem_t) * n_patches * patch_size);
  memset(seq2col_out, 0, sizeof(elem_t) * n_patches * patch_size);

  weight = (elem_t *)malloc(sizeof(elem_t) * groups * patch_size * 1);
  bias = (elem_t *)malloc(sizeof(elem_t) * out_features);

  assert(params.size() == DEPTHWISECONV1D_NUM_PARAMS);

  memcpy(weight, params[WEIGHT_IDX], sizeof(elem_t) * groups * patch_size * 1);
  memcpy(bias, params[BIAS_IDX], sizeof(elem_t) * out_features);

  assert(scales.size() == DEPTHWISECONV1D_NUM_SCALES);
  output_scale = scales[0];

  return 0;
}

int DepthwiseConv1d::seq2col(const elem_t *input) {
  int patch_base_row = 0;
  int patches_per_group = batch_size * out_length;

  // iterate over batch
  for (int n_batch = 0; n_batch < batch_size; n_batch++) {
    // iterate over image coordinates
    for (int im_col = -padding; im_col < in_length - kernel_reach + 1;
         im_col += stride) {
      // iterate over filter coordinates
      int patch_col = 0;
      for (int filter_col = 0; filter_col < kernel_size; filter_col++) {
        int pixel_col = im_col + (filter_col * dilation);

        for (int im_channel = 0; im_channel < in_features; im_channel++) {
          int patch_row = patch_base_row + im_channel * patches_per_group;

          if (pixel_col < 0 || pixel_col >= in_length) {
            // output[patch_row][patch_col] = 0;
          }
          else {
            seq2col_out[ARRAY_INDEX_2D(patch_size, patch_row, patch_col)] =
                input[ARRAY_INDEX_3D(in_length, in_features, n_batch, pixel_col,
                                     im_channel)];
          }
        }
        patch_col++;
      }

      patch_base_row++;
    }
  }

  return 0;
}

int DepthwiseConv1d::forward(elem_t *input, elem_t *output) {
#if PROFILE
  auto start = read_cycles();
#endif
  // seq2col
  seq2col(input);
#if PROFILE
  auto seq2col = read_cycles();
#endif

  // matmul
  int col_stride = batch_size * out_length * kernel_size;
  int weight_stride = patch_size;
  int bias_stride = 1;
  int output_stride = batch_size * out_length;

  for (int i = 0; i < groups; i++) {
    int col_offset = i * col_stride;
    int weight_offset = i * weight_stride;
    int bias_offset = i * bias_stride;
    int output_offset = i * output_stride;

    tiled_matmul_auto(n_patches / groups, 1, patch_size,

                      seq2col_out + col_offset, weight + weight_offset,
                      bias + bias_offset, (elem_t *)output + output_offset,

                      patch_size, 1, 1, 1, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
                      MVIN_SCALE_ONE, activation, 0, 0, true, CPU);
  }

#if PROFILE
  auto matmul = read_cycles();
  printf("dwconv %d %d\n", seq2col - start, matmul - seq2col);
#endif

  return 0;
}

void DepthwiseConv1d::print_weights() {
  print_mat3d("weight", weight, in_features, kernel_size, 1);
  print_vec1d("bias", bias, in_features);
}

DWSConv1d::DWSConv1d(int batch_size, int in_features, int out_features,
                     int in_length, int kernel_size, int stride, int dilation,
                     int padding, int activation, bool has_bias)
    : Layer(batch_size, in_features, out_features),
      convs{(Conv1d *)nullptr, (Conv1d *)nullptr} {
  convs[0] = new DepthwiseConv1d(batch_size, in_features, in_features,

                                 in_length, kernel_size, stride, dilation,
                                 padding, activation, has_bias);
  out_length = convs[0]->get_out_length();
  convs[1] = new PointwiseConv1d(batch_size, in_features, out_features,
                                 in_length, activation, has_bias);
}

DWSConv1d::~DWSConv1d() {
  for (auto conv : convs)
    delete conv;

  delete hidden;
}

int DWSConv1d::prepare() {
  hidden =
      (elem_t *)malloc(sizeof(elem_t) * batch_size * out_length * in_features);

  for (auto conv : convs) {
    conv->prepare();
  }

  return 0;
}

int DWSConv1d::prepare(std::vector<elem_t *> &params,
                       std::vector<int> &scales) {
  hidden =
      (elem_t *)malloc(sizeof(elem_t) * batch_size * out_length * in_features);

  assert(!params.empty());
  assert(!scales.empty());

  auto begin_param = params.begin();
  auto begin_scale = scales.begin();
  for (int i = 0; i < convs.size(); i++) {
    std::vector<elem_t *> subparam(begin_param + i * CONV1D_NUM_PARAMS,
                                   begin_param + (i + 1) * CONV1D_NUM_PARAMS);
    std::vector<int> subscale(begin_scale + i * CONV1D_NUM_SCALES,
                              begin_scale + (i + 1) * CONV1D_NUM_SCALES);

    convs[i]->prepare(subparam, subscale);
  }
  return 0;
}

int DWSConv1d::forward(elem_t *input, elem_t *output) {
  convs[0]->forward(input, hidden);
  convs[1]->forward(hidden, output);

  return 0;
}

void DWSConv1d::print_weights() {
  printf("depthwise\n");
  convs[0]->print_weights();
  printf("pointwise\n");
  convs[1]->print_weights();
}
