// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include "interval_time_model_test.hh"

#include <cinttypes>

int main() {
  printf("Running Interval Time Model Test..\n");

  // conversion
  f32MatTof16((ushort *)input, (float *)input_fp32,
              BATCH_SIZE * IN_LENGTH * IN_FEATURES);

  f32MatTof16((ushort *)weight1, (float *)weight1_fp32,
              HIDDEN_FEATURES * KERNEL_SIZE * IN_FEATURES);
  f32MatTof16((ushort *)bias1, (float *)bias1_fp32, HIDDEN_FEATURES);

  f32MatTof16((ushort *)weight2, (float *)weight2_fp32,
              HIDDEN_FEATURES * KERNEL_SIZE * HIDDEN_FEATURES);
  f32MatTof16((ushort *)bias2, (float *)bias2_fp32, HIDDEN_FEATURES);

  f32MatTof16((ushort *)weight1x1, (float *)weight1x1_fp32,
              HIDDEN_FEATURES * 1 * IN_FEATURES);
  f32MatTof16((ushort *)bias1x1, (float *)bias1x1_fp32, HIDDEN_FEATURES);

  f32MatTof16((ushort *)weight_linear, (float *)weight_linear_fp32,
              HIDDEN_FEATURES * OUT_CATEGORIES);
  f32MatTof16((ushort *)bias_linear, (float *)bias_linear_fp32, OUT_CATEGORIES);

  f32MatTof16((ushort *)output, (float *)output_fp32,
              BATCH_SIZE * OUT_CATEGORIES);

#if DEBUG
  printf("Params:\n");
  printf(" DIM:             %8d\n", DIM);
  printf(" ADDR_LEN:        %8d\n", ADDR_LEN);
  printf(" BANK_NUM:        %8d\n", BANK_NUM);
  printf(" BANK_ROWS:       %8d\n", BANK_ROWS);
  printf(" ACC_ROWS:        %8d\n", ACC_ROWS);
  printf(" MAX_BYTES:       %8d\n", MAX_BYTES);
  printf(" BATCH_SIZE:      %8d\n", BATCH_SIZE);
  printf(" IN_FEATURES:     %8d\n", IN_FEATURES);
  printf(" HIDDEN_FEATURES: %8d\n", HIDDEN_FEATURES);
  printf(" OUT_CATEGORIES:  %8d\n", OUT_CATEGORIES);
  printf(" IN_LENGTH:       %8d\n", IN_LENGTH);
  printf(" NUM_BLOCKS:      %8d\n", NUM_BLOCKS);
  printf(" KERNEL_SIZE:     %8d\n", KERNEL_SIZE);
  printf(" MODE:            %8d\n", MODE);
  printf(" MATMUL_TYPE:     %8s\n", get_matmul_type_name(MATMUL_TYPE));
#endif

#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  gemmini_flush(0);
#if DEBUG
  printf("Output dimension: %u\n\n", IN_LENGTH);

  printf("creating model..\n");
#endif

  IntervaltimeModel mymodel(BATCH_SIZE, IN_FEATURES, HIDDEN_FEATURES,
                            OUT_CATEGORIES, IN_LENGTH, NUM_BLOCKS, KERNEL_SIZE,
                            (Mode)MODE, MATMUL_TYPE);

#if DEBUG
  printf("preparing model..\n");
  std::vector<elem_t *> model_params{
      (elem_t *)weight1,       (elem_t *)bias1,      (elem_t *)weight2,
      (elem_t *)bias2,         (elem_t *)weight1x1,  (elem_t *)bias1x1,
      (elem_t *)weight_linear, (elem_t *)bias_linear};
  std::vector<int> model_scale{0, 1, 1, 1, 1};
  mymodel.prepare(model_params, model_scale);
#else
  mymodel.prepare();
#endif

#if DEBUG
  printf("Gemmini inference...\n");
#endif
  uint64_t start_gemmini = read_cycles();

  mymodel.forward((elem_t *)input, (elem_t *)output_mat);

  uint64_t end_gemmini = read_cycles();

  auto &ticks = mymodel.getTicks();

#if DEBUG
  printf("Cycle information:\n");

  for (int i = 0; i < ticks.size(); i++) {
    uint64_t begin = i == 0 ? start_gemmini : ticks[i - 1];
    uint64_t t = ticks[i] - begin;

    printf(" Layer : %" PRIu64 "\n", t);
  }

  printf(" Total : %" PRIu64 "\n", end_gemmini - start_gemmini);
#else
  printf("%d\t%" PRIu64 "\t", NUM_BLOCKS, start_gemmini);

  for (auto &iter : ticks) {
    printf("%" PRIu64 "\t", iter);
  }

  printf("\n");

#endif
  assert(sizeof(output_mat) == sizeof(output));

  bool success = is_vec_equal(&output[0][0], &output_mat[0][0],
                              sizeof(output) / sizeof(elem_t));

#if DEBUG
  if (success) {
    printf("SUCCESS!\n");
  }
  else {
    printf("FAILED!\n");

    mymodel.print_weights();

    print2dmatrix("output", (elem_t *)output, BATCH_SIZE, OUT_CATEGORIES);

    print2dmatrix("output_mat", (elem_t *)output_mat, BATCH_SIZE,
                  OUT_CATEGORIES);

    return 1;
  }
#endif

  return 0;
}
