// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */
#include "gemm.hh"

void gemm_cpu(float *OUT, float *A, float *B, float *C, const int64_t i,
                   const int64_t j, const int64_t k) {
  for (int64_t _i = 0; _i < i; _i++) {
    for (int64_t _j = 0; _j < j; _j++) {
      for (int64_t _k = 0; _k < k; _k++) {
        OUT[_i * k + _k] += A[_i * j + _j] * B[_j * k + _k];
      }
    }

    if (C != nullptr) {
      for (int64_t _k = 0; _k < k; _k++) {
        OUT[_i * k + _k] += C[_k];
      }
    }
  }
}
