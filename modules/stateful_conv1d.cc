// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include "stateful_conv1d.hh"

StatefulConv1d::StatefulConv1d(int batch_size, int in_features,
                               int out_features, int in_length, int kernel_size,
                               int stride, int dilation, int padding,
                               int activation, bool has_bias, tiled_matmul_type_t matmul_type)
    : Conv1d(batch_size, in_features, out_features, in_length, kernel_size,
             stride, dilation, padding, activation, has_bias, matmul_type) {
  assert(stride == 1);
  n_patches = batch_size;
}

StatefulConv1d::~StatefulConv1d() {
  free(old_in);
  free(in_merged);
}

int StatefulConv1d::prepare() {
  old_in = (elem_t *)malloc(sizeof(elem_t) * batch_size * in_features *
                            (kernel_reach - 1));
  in_merged = (elem_t *)malloc(sizeof(elem_t) * batch_size * in_features *
                               kernel_reach);
  memset(old_in, 0,
         sizeof(elem_t) * batch_size * in_features * (kernel_reach - 1));

  Conv1d::prepare();

  return 0;
}

int StatefulConv1d::prepare(std::vector<elem_t *> &params,
                            std::vector<int> &scales) {
  old_in = (elem_t *)malloc(sizeof(elem_t) * batch_size * in_features *
                            (kernel_reach - 1));
  in_merged = (elem_t *)malloc(sizeof(elem_t) * batch_size * in_features *
                               kernel_reach);
  memset(old_in, 0,
         sizeof(elem_t) * batch_size * in_features * (kernel_reach - 1));

  Conv1d::prepare(params, scales);

  return 0;
}

int StatefulConv1d::seq2col(const elem_t *input) {
  // do seq2col only for last time index
  int im_col = 0;

  int patch_row = 0;
  // iterate over batch
  for (int n_batch = 0; n_batch < batch_size; n_batch++) {

    int patch_col = 0;
    // iterate over filter coordinates
    for (int filter_col = 0; filter_col < kernel_size; filter_col++) {
      for (int im_channel = 0; im_channel < in_features; im_channel++) {
        int pixel_col = im_col + filter_col * dilation;

        if (pixel_col < 0) {
          // output[patch_row][patch_col] = 0;
        } else {
          seq2col_out[ARRAY_INDEX_2D(patch_size, patch_row, patch_col)] =
              input[ARRAY_INDEX_3D(kernel_reach, in_features, n_batch,
                                   pixel_col, im_channel)];
        }

        patch_col++;
      }
    }

    patch_row++;
  }
  return 0;
}

int StatefulConv1d::merge_and_save(elem_t *new_input) {

  for (int b = 0; b < batch_size; b++) {
    // merge old_out & new_out
    // old_in
    for (int l = 0; l < kernel_reach - 1; l++) {
      for (int f = 0; f < in_features; f++) {
        in_merged[ARRAY_INDEX_3D(kernel_reach, in_features, b, l, f)] =
            old_in[ARRAY_INDEX_3D(kernel_reach - 1, in_features, b, l, f)];
      }
    }

    // new_input
    for (int f = 0; f < in_features; f++) {
      in_merged[ARRAY_INDEX_3D(kernel_reach, in_features, b, kernel_reach - 1,
                               f)] =
          new_input[ARRAY_INDEX_2D(in_features, b, f)];
    }

    // save current input to old_in
    for (int l = 1; l < kernel_reach; l++) {
      for (int f = 0; f < in_features; f++) {
        old_in[ARRAY_INDEX_3D(kernel_reach - 1, in_features, b, l - 1, f)] =
            in_merged[ARRAY_INDEX_3D(kernel_reach, in_features, b, l, f)];
      }
    }
  }

  return 0;
}

int StatefulConv1d::forward(elem_t *input, elem_t *output) {

  merge_and_save(input);

  seq2col(in_merged);

  tiled_matmul_auto(n_patches, out_features, patch_size,

                    seq2col_out, weight, bias, output, patch_size, out_features,

                    out_features, out_features, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
                    MVIN_SCALE_ONE, activation, 0, 0, true, matmul_type);

  return 0;
}

// MUST DO WARMUP before forward for correct computation
int StatefulConv1d::warmup(elem_t *input, elem_t *output) {
  // temporarily store original seq2col_out
  elem_t *original_seq2col_out = seq2col_out;

  seq2col_out = (elem_t *)malloc(sizeof(elem_t) * batch_size * out_length *
                                 kernel_size * in_features);

  // save current output to old_out
  for (int b = 0; b < batch_size; b++) {
    for (int l = in_length - kernel_reach + 1; l < in_length; l++) {
      for (int f = 0; f < in_features; f++) {
        old_in[ARRAY_INDEX_3D(kernel_reach - 1, in_features, b,
                              l - (in_length - kernel_reach + 1), f)] =
            input[ARRAY_INDEX_3D(in_length, in_features, b, l, f)];
      }
    }
  }

  // same with original Conv1d::forward
  Conv1d::seq2col(input);
  tiled_matmul_auto(batch_size * out_length, out_features, patch_size,
                    seq2col_out, weight, bias, output, patch_size, out_features,
                    out_features, out_features, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
                    MVIN_SCALE_ONE, activation, 0, 0, true, matmul_type);

  // restore seq2col_out
  free(seq2col_out);
  seq2col_out = original_seq2col_out;

  return 0;
}
