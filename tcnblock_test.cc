// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include "tcnblock_test.hh"

int main() {
  printf("Running TCN Block Test..\n");
#if !defined(BAREMETAL) && !defined(SPIKE)
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  f32MatTof16((ushort *)input, (float *)input_fp32,
              BATCH_SIZE * IN_LENGTH * IN_FEATURES);

  f32MatTof16((ushort *)weight1, (float *)weight1_fp32,
              OUT_FEATURES * KERNEL_SIZE * IN_FEATURES);
  f32MatTof16((ushort *)bias1, (float *)bias1_fp32, OUT_FEATURES);

  f32MatTof16((ushort *)weight2, (float *)weight2_fp32,
              OUT_FEATURES * KERNEL_SIZE * OUT_FEATURES);
  f32MatTof16((ushort *)bias2, (float *)bias2_fp32, OUT_FEATURES);

  f32MatTof16((ushort *)weight1x1, (float *)weight1x1_fp32,
              OUT_FEATURES * 1 * IN_FEATURES);
  f32MatTof16((ushort *)bias1x1, (float *)bias1x1_fp32, OUT_FEATURES);
  f32MatTof16((ushort *)output, (float *)output_fp32,
              BATCH_SIZE * OUT_FEATURES);

  gemmini_flush(0);

  printf("Output dimension: %u\n\n", IN_LENGTH);

  printf("creating TCNBlock..\n");
  std::pair<int, int> dilation(DILATION1, DILATION2);
  OriginalTCNBlock mytcnblock(BATCH_SIZE, IN_FEATURES, OUT_FEATURES, IN_LENGTH,
                              KERNEL_SIZE, STRIDE, dilation, RELU, !NO_BIAS);

  std::vector<elem_t *> tcn_params{(elem_t *)weight1,   (elem_t *)bias1,
                                   (elem_t *)weight2,   (elem_t *)bias2,
                                   (elem_t *)weight1x1, (elem_t *)bias1x1};
  std::vector<int> tcn_scale{0, 1, 1, 1};
  mytcnblock.prepare(tcn_params, tcn_scale);

  printf("Gemmini conv...\n");
  uint64_t start_gemmini = read_cycles();

  mytcnblock.forward((elem_t *)input, (elem_t *)output_mat);

  uint64_t end_gemmini = read_cycles();
  printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);

  assert(sizeof(output_mat) == sizeof(output));

  bool success = is_vec_equal(&output[0][0][0], &output_mat[0][0],
                              sizeof(output) / sizeof(elem_t));

  if (success) {
    printf("SUCCESS!\n");
  }
  else {
    printf("FAILED!\n");

    mytcnblock.print_weights();

    print_mat3d("input", (elem_t *)input, BATCH_SIZE, IN_LENGTH, IN_FEATURES);

    print_mat3d("output", (elem_t *)output, BATCH_SIZE, IN_LENGTH,
                  OUT_FEATURES);

    print_vec2d("output_mat", (elem_t *)output_mat, BATCH_SIZE * IN_LENGTH,
                  OUT_FEATURES);

    return 1;
  }

  return 0;
}
