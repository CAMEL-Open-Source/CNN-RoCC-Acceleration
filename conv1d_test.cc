// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "gemmini-api/gemmini_testutils.h"
#include "modules/layer.hh"
#include "util/util.hh"

#define BATCH_SIZE 1
#define IN_LENGTH 8
#define IN_FEATURES 3
#define OUT_FEATURES 4
#define KERNEL_DIM 3
#define PADDING 4
#define STRIDE 1
#define DILATION 2

#define NO_BIAS false

#define OUT_LENGTH ((IN_LENGTH + PADDING - KERNEL_DIM) / STRIDE + 1)
#define PATCH_SIZE (KERNEL_DIM * IN_FEATURES)
#define N_PATCHES (BATCH_SIZE * OUT_LENGTH)

bool vec_is_equal(elem_t *a, elem_t *b, int len) {
  for (int i = 0; i < len; i++)
    if (a[i] != b[i])
      return false;
  return true;
}

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  gemmini_flush(0);

  // assert((IN_LENGTH + 2*padding - kernel_dim) % stride == 0);

  printf("Output dimension: %u\n\n", OUT_LENGTH);

  static float input_fp32[BATCH_SIZE][IN_LENGTH][IN_FEATURES] = {{{2, 3, 4},
                                                                  {4, 2, 0},
                                                                  {2, 1, 4},
                                                                  {2, 1, 0},
                                                                  {0, 4, 0},
                                                                  {0, 4, 4},
                                                                  {3, 4, 1},
                                                                  {2, 4, 3}}};
  static float weights_fp32[OUT_FEATURES][KERNEL_DIM][IN_FEATURES] = {
      {{1, 1, 0}, {1, 0, 0}, {1, 2, 3}},

      {{1, 1, 1}, {2, 2, 3}, {3, 3, 2}},

      {{2, 4, 1}, {4, 4, 1}, {1, 3, 0}},

      {{3, 3, 0}, {1, 3, 0}, {4, 3, 1}}};
  static float bias_fp32[OUT_FEATURES] = {1, 0, 1, 0};
  static float output_fp32[BATCH_SIZE][OUT_LENGTH][OUT_FEATURES] = {
      {{21, 23, 12, 21},
       {9, 18, 11, 22},
       {19, 39, 30, 26},
       {9, 21, 30, 21},
       {16, 39, 49, 32},
       {29, 32, 41, 39},
       {18, 38, 44, 46},
       {23, 47, 43, 44}}};

  static elem_t input[BATCH_SIZE][IN_LENGTH][IN_FEATURES];
  static elem_t weights[OUT_FEATURES][KERNEL_DIM][IN_FEATURES];
  static acc_t bias[OUT_FEATURES];
  static elem_t output[BATCH_SIZE][OUT_LENGTH][OUT_FEATURES];
  static elem_t output_mat[N_PATCHES][OUT_FEATURES];

  f32MatTof16((ushort *)input, (float *)input_fp32,
              BATCH_SIZE * IN_LENGTH * IN_FEATURES);
  f32MatTof16((ushort *)weights, (float *)weights_fp32,
              OUT_FEATURES * KERNEL_DIM * IN_FEATURES);
  f32MatTof16((ushort *)bias, (float *)bias_fp32, OUT_FEATURES);
  f32MatTof16((ushort *)output, (float *)output_fp32,
              BATCH_SIZE * OUT_LENGTH * OUT_FEATURES);

  printf("creating Conv layer..\n");
  Conv1d myconv(BATCH_SIZE, IN_FEATURES, OUT_FEATURES, IN_LENGTH, KERNEL_DIM,
                STRIDE, DILATION, PADDING, NO_ACTIVATION, !NO_BIAS);
  std::vector<elem_t *> conv_params{(elem_t *)weights, (elem_t *)bias};
  std::vector<int> conv_scale(1, 0);
  myconv.prepare(conv_params, conv_scale);

  printf("Gemmini conv...\n");
  uint64_t start_gemmini = read_cycles();

  myconv.forward((elem_t *)input, (elem_t *)output_mat);

  uint64_t end_gemmini = read_cycles();
  printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);

  assert(sizeof(output_mat) == sizeof(output));

  bool success = vec_is_equal(&output[0][0][0], &output_mat[0][0],
                              sizeof(output) / sizeof(elem_t));

  if (success) {
    printf("SUCCESS!\n");
  }
  else {
    printf("FAILED!\n");

    myconv.print_weights();

    print3dmatrix("input", (elem_t *)input, BATCH_SIZE, IN_LENGTH, IN_FEATURES);

    print3dmatrix("output", (elem_t *)output, BATCH_SIZE, OUT_LENGTH,
                  OUT_FEATURES);

    print2dmatrix("output_mat", (elem_t *)output_mat, BATCH_SIZE * OUT_LENGTH,
                  OUT_FEATURES);

    return 1;
  }

  return 0;
}
