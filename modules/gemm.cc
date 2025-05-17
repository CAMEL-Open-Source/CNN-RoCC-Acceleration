/**
 * gemm.cc
 * Copyright (C) 2021-2024 CAMELab
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include "gemm.hh"

void gemm_cpu_fp32(float *OUT, float *A, float *B, float *C, const int64_t i,
                   const int64_t j, const int64_t k) {
  for (int64_t _i = 0; _i < i; _i++) {
    // multiplication and accumulation
    for (int64_t _j = 0; _j < j; _j++) {
      for (int64_t _k = 0; _k < k; _k++) {
        OUT[_i * k + _k] += A[_i * j + _j] * B[_j * k + _k];
      }
    }

    // bias
    if (C != nullptr) {
      for (int64_t _k = 0; _k < k; _k++) {
        OUT[_i * k + _k] += C[_k];
      }
    }
  }
}

// template instantiations
#define INSTANTIATE_GEMM_FP(device, type)                                      \
  template void gemm_##device<type>(type *, type *, type *, type *,            \
                                    const int64_t, const int64_t,              \
                                    const int64_t);
INSTANTIATE_GEMM_FP(cpu, float)
