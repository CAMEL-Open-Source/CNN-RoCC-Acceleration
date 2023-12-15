// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include "dws_conv1d_test.hh"

int main() {
  printf("Running Depthwise Convolution 1D Test..\n");
#if !defined(BAREMETAL) && !defined(SPIKE)
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  f32MatTof16((ushort *)input, (float *)input_fp32,
              BATCH_SIZE * IN_LENGTH * IN_FEATURES);

  f32MatTof16((ushort *)dw_weight, (float *)dw_weight_fp32,
              IN_FEATURES * KERNEL_SIZE * 1);
  f32MatTof16((ushort *)dw_bias, (float *)dw_bias_fp32, IN_FEATURES);

  f32MatTof16((ushort *)pw_weight, (float *)pw_weight_fp32,
              OUT_FEATURES * 1 * IN_FEATURES);
  f32MatTof16((ushort *)pw_bias, (float *)pw_bias_fp32, OUT_FEATURES);

  f32MatTof16((ushort *)output, (float *)output_fp32,
              BATCH_SIZE * IN_LENGTH * OUT_FEATURES);

  gemmini_flush(0);

  printf("Output dimension: %u\n\n", IN_LENGTH);

  printf("creating dwsconv1d..\n");
  DWSConv1d mydwsconv(BATCH_SIZE, IN_FEATURES, OUT_FEATURES, IN_LENGTH,
                      KERNEL_SIZE, STRIDE, DILATION, PADDING, RELU, BIAS);

  std::vector<elem_t *> dwsconv_params{(elem_t *)dw_weight, (elem_t *)dw_bias,
                                       (elem_t *)pw_weight, (elem_t *)pw_bias};
  std::vector<int> dwsconv_scale{1, 1};
  mydwsconv.prepare(dwsconv_params, dwsconv_scale);
  mydwsconv.print_weights();

  printf("Gemmini conv...\n");
  uint64_t start_gemmini = read_cycles();

  mydwsconv.forward((elem_t *)input, (elem_t *)output_mat);

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

    mydwsconv.print_weights();

    print_mat3d("input", (elem_t *)input, BATCH_SIZE, IN_LENGTH, IN_FEATURES);

    print_mat3d("output", (elem_t *)output, BATCH_SIZE, IN_LENGTH,
                  OUT_FEATURES);

    print_mat2d("output_mat", (elem_t *)output_mat, BATCH_SIZE * IN_LENGTH,
                  OUT_FEATURES);

    return 1;
  }

  return 0;
}
