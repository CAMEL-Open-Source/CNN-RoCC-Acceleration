// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include "layer.hh"

Layer::Layer(int b, int inf, int outf)
    : batch_size(b), in_features(inf), out_features(outf) {}

Conv1d::Conv1d(int batch_size, int in_features, int out_features, int in_length,
               int kernel_size, int stride, int dilation, int padding,
               int activation, bool has_bias, tiled_matmul_type_t matmul_type)
    : Layer(batch_size, in_features, out_features) {
  this->in_length = in_length;
  this->kernel_size = kernel_size;
  this->stride = stride;
  this->dilation = dilation;
  this->padding = padding;

  this->activation = activation;
  this->has_bias = has_bias;

  // below is "in_length + 2*padding" in conventional conv1d
  int input_with_padding = in_length + padding;
  this->kernel_reach = dilation * (kernel_size - 1) + 1;
  assert(input_with_padding > kernel_reach);

  this->out_length = (input_with_padding - kernel_reach) / stride + 1;
  this->n_patches = batch_size * out_length;
  this->patch_size = kernel_size * in_features;

  this->matmul_type = matmul_type;
}

Conv1d::~Conv1d() {
  free(seq2col_out);
  free(weight);
  free(bias);
}

int Conv1d::prepare() {
  seq2col_out = (elem_t *)malloc(sizeof(elem_t) * n_patches * patch_size);
  weight = (elem_t *)malloc(sizeof(elem_t) * patch_size * out_features);
  bias = (elem_t *)malloc(sizeof(elem_t) * out_features);

  memset(seq2col_out, 0, sizeof(elem_t) * n_patches * patch_size);
  memset(weight, 0, sizeof(elem_t) * patch_size * out_features);
  memset(bias, 0, sizeof(elem_t) * out_features);

  output_scale = 0;

  return 0;
}

int Conv1d::prepare(std::vector<elem_t *> &params, std::vector<int> &scales) {
  seq2col_out = (elem_t *)malloc(sizeof(elem_t) * n_patches * patch_size);
  memset(seq2col_out, 0, sizeof(elem_t) * n_patches * patch_size);

  weight = (elem_t *)malloc(sizeof(elem_t) * patch_size * out_features);
  bias = (elem_t *)malloc(sizeof(elem_t) * out_features);

  assert(params.size() == CONV1D_NUM_PARAMS);
  for (int outc = 0; outc < out_features; outc++) {
    for (int kcol = 0; kcol < kernel_size; kcol++) {
      for (int inc = 0; inc < in_features; inc++) {
        int wmatrow = kcol * in_features + inc;
        weight[ARRAY_INDEX_2D(out_features, wmatrow, outc)] =
            params[WEIGHT_IDX]
                  [ARRAY_INDEX_3D(kernel_size, in_features, outc, kcol, inc)];
      }
    }
  }
  memcpy(bias, params[BIAS_IDX], sizeof(elem_t) * out_features);

  assert(scales.size() == CONV1D_NUM_SCALES);
  output_scale = scales[0];

  return 0;
}

int Conv1d::seq2col(const elem_t *input) {
  int patch_row = 0;
  // iterate over batch
  for (int n_batch = 0; n_batch < batch_size; n_batch++) {
    // iterate over image coordinates
    for (int im_col = -padding; im_col < in_length - kernel_reach + 1;
         im_col += stride) {
      // iterate over filter coordinates
      int patch_col = 0;
      for (int filter_col = 0; filter_col < kernel_size; filter_col++) {
        for (int im_channel = 0; im_channel < in_features; im_channel++) {
          int pixel_col = im_col + (filter_col * dilation);

          if (pixel_col < 0 || pixel_col >= in_length) {
            // output[patch_row][patch_col] = 0;
          }
          else {
            seq2col_out[ARRAY_INDEX_2D(patch_size, patch_row, patch_col)] =
                input[ARRAY_INDEX_3D(in_length, in_features, n_batch, pixel_col,
                                     im_channel)];
          }

          patch_col++;
        }
      }

      patch_row++;
    }
  }
  return 0;
}

int Conv1d::forward(elem_t *input, elem_t *output) {
  seq2col(input);

  tiled_matmul_auto(n_patches, out_features, patch_size, seq2col_out, weight,
                    bias, output, patch_size, out_features, out_features,
                    out_features, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
                    MVIN_SCALE_ONE, activation, 0, 0, true, matmul_type);

  return 0;
}

void Conv1d::print_weights() {
  print2dmatrix("conv1d weight", weight, patch_size, out_features);
  print1dvector("conv1d bias", bias, out_features);
}

Linear::Linear(int batch_size, int in_features, int out_features, bool has_bias,
               int activation, tiled_matmul_type_t matmul_type)
    : Layer(batch_size, in_features, out_features) {
  this->has_bias = has_bias;
  this->activation = activation;
  this->matmul_type = matmul_type;
}

Linear::~Linear() {
  free(weight);
  free(bias);
}

int Linear::prepare() {
  weight = (elem_t *)malloc(sizeof(elem_t) * in_features * out_features);
  bias = (elem_t *)malloc(sizeof(elem_t) * out_features);

  memset(weight, 0, sizeof(elem_t) * in_features * out_features);
  memset(bias, 0, sizeof(elem_t) * out_features);

  output_scale = 0;

  return 0;
}

int Linear::prepare(std::vector<elem_t *> &params, std::vector<int> &scales) {
#if DEBUG
  printf("linear prepare\n");
#endif

  weight = (elem_t *)malloc(sizeof(elem_t) * in_features * out_features);
  bias = (elem_t *)malloc(sizeof(elem_t) * out_features);

  assert(params.size() == LINEAR_NUM_PARAMS);
  memcpy(weight, params[WEIGHT_IDX],
         sizeof(elem_t) * in_features * out_features);
  memcpy(bias, params[BIAS_IDX], sizeof(elem_t) * out_features);

  assert(scales.size() == LINEAR_NUM_SCALES);
  output_scale = scales[0];

  return 0;
}

int Linear::forward(elem_t *input, elem_t *output) {
  tiled_matmul_auto(batch_size, out_features, in_features, input, weight, bias,
                    output, in_features, out_features, out_features,
                    out_features, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
                    MVIN_SCALE_ONE, activation, 0, 0, true, matmul_type);

  return 0;
}

// TODO
int Linear::backward(elem_t *doutput, elem_t *dinput) {
  // doutput/dweight

  // doutput/dbias

  // doutput/dinput

  return 0;
}

void Linear::print_weights() {
  printf("Linear\n");
  print2dmatrix("weight", weight, in_features, out_features);
  print1dvector("bias", bias, out_features);
}
