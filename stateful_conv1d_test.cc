#include "gemmini-api/gemmini_testutils.h"
#include "modules/stateful_conv1d.hh"
#include "util/util.hh"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#define BATCH_SIZE 1
#define IN_LENGTH 8
#define IN_FEATURES 3
#define OUT_FEATURES 4
#define KERNEL_DIM 3
#define PADDING 2
#define STRIDE 1
#define DILATION 1

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

  static float input1_fp32[BATCH_SIZE][IN_LENGTH][IN_FEATURES] = {{{4, 1, 1},
                                                                   {4, 1, 4},
                                                                   {3, 1, 1},
                                                                   {0, 4, 4},
                                                                   {3, 3, 4},
                                                                   {4, 1, 1},
                                                                   {2, 1, 4},
                                                                   {3, 3, 4}}};

  static float input2_fp32[BATCH_SIZE][1][IN_FEATURES] = {{{2, 4, 4}}};
  static float input3_fp32[BATCH_SIZE][1][IN_FEATURES] = {{{3, 3, 0}}};

  static float weights_fp32[OUT_FEATURES][KERNEL_DIM][IN_FEATURES] = {
      {{1, 0, 2}, {2, 0, 4}, {3, 0, 1}},

      {{3, 1, 2}, {3, 4, 3}, {3, 1, 0}},

      {{2, 4, 1}, {1, 3, 0}, {0, 1, 3}},

      {{1, 4, 4}, {1, 1, 0}, {2, 3, 0}}};

  static float bias_fp32[OUT_FEATURES] = {3, 2, 3, 2};

  static float output1_fp32[BATCH_SIZE][OUT_LENGTH][OUT_FEATURES] = {
      {{16, 15, 7, 13},
       {31, 34, 23, 18},
       {43, 55, 27, 28},
       {29, 43, 41, 42},
       {37, 54, 41, 32},
       {46, 60, 39, 51},
       {36, 48, 45, 45},
       {42, 51, 36, 32}}};

  static float output2_fp32[BATCH_SIZE][1][OUT_FEATURES] = {{{45, 60, 43, 46}}};
  static float output3_fp32[BATCH_SIZE][1][OUT_FEATURES] = {{{43, 68, 42, 54}}};

  static elem_t input1[BATCH_SIZE][IN_LENGTH][IN_FEATURES];
  static elem_t input2[BATCH_SIZE][1][IN_FEATURES];
  static elem_t input3[BATCH_SIZE][1][IN_FEATURES];
  static elem_t weights[OUT_FEATURES][KERNEL_DIM][IN_FEATURES];
  static acc_t bias[OUT_FEATURES];
  static elem_t output1[BATCH_SIZE][OUT_LENGTH][OUT_FEATURES];
  static elem_t output2[BATCH_SIZE][1][OUT_FEATURES];
  static elem_t output3[BATCH_SIZE][1][OUT_FEATURES];
  static elem_t output_mat[BATCH_SIZE][OUT_LENGTH][OUT_FEATURES];

  f32MatTof16((ushort *)input1, (float *)input1_fp32,
              BATCH_SIZE * IN_LENGTH * IN_FEATURES);
  f32MatTof16((ushort *)input2, (float *)input2_fp32, BATCH_SIZE * IN_FEATURES);
  f32MatTof16((ushort *)input3, (float *)input3_fp32, BATCH_SIZE * IN_FEATURES);

  f32MatTof16((ushort *)weights, (float *)weights_fp32,
              OUT_FEATURES * KERNEL_DIM * IN_FEATURES);
  f32MatTof16((ushort *)bias, (float *)bias_fp32, OUT_FEATURES);

  f32MatTof16((ushort *)output1, (float *)output1_fp32,
              BATCH_SIZE * OUT_LENGTH * OUT_FEATURES);
  f32MatTof16((ushort *)output2, (float *)output2_fp32,
              BATCH_SIZE * OUT_FEATURES);
  f32MatTof16((ushort *)output3, (float *)output3_fp32,
              BATCH_SIZE * OUT_FEATURES);

  printf("creating Conv layer..\n");
  StatefulConv1d myconv(BATCH_SIZE, IN_FEATURES, OUT_FEATURES, IN_LENGTH,
                        KERNEL_DIM, STRIDE, DILATION, PADDING, NO_ACTIVATION,
                        !NO_BIAS);
  std::vector<elem_t *> conv_params{(elem_t *)weights, (elem_t *)bias};
  std::vector<int> conv_scale(1, 0);
  myconv.prepare(conv_params, conv_scale);

  printf("Warming up..\n");
  myconv.warmup((elem_t *)input1, (elem_t *)output_mat);
  print2dmatrix("output_mat1", (elem_t *)output_mat, BATCH_SIZE * OUT_LENGTH,
                OUT_FEATURES);
  bool success = vec_is_equal(&output1[0][0][0], (elem_t *)output_mat,
                              sizeof(output1) / sizeof(elem_t));
  if (!success)
    printf("FAIL\n");

  printf("Gemmini conv...\n");
  uint64_t start_gemmini = read_cycles();

  myconv.forward((elem_t *)input2, (elem_t *)output_mat);

  uint64_t end_gemmini = read_cycles();
  printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);
  print2dmatrix("output_mat2", (elem_t *)output_mat, BATCH_SIZE, OUT_FEATURES);

  success = success && vec_is_equal(&output2[0][0][0], (elem_t *)output_mat,
                                    sizeof(output2) / sizeof(elem_t));
  if (!success)
    printf("FAIL\n");

  printf("Gemmini conv...\n");
  start_gemmini = read_cycles();

  myconv.forward((elem_t *)input3, (elem_t *)output_mat);

  end_gemmini = read_cycles();
  printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);
  print2dmatrix("output_mat3", (elem_t *)output_mat, BATCH_SIZE, OUT_FEATURES);

  success = success && vec_is_equal(&output3[0][0][0], (elem_t *)output_mat,
                                    sizeof(output3) / sizeof(elem_t));
  if (!success)
    printf("FAIL\n");

  if (success) {
    printf("SUCCESS!\n");
  } else {
    printf("FAILED!\n");

    myconv.print_weights();

    return 1;
  }

  return 0;
}
