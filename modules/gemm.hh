// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#ifndef __SRC_LAYERS_GEMM_H__
#define __SRC_LAYERS_GEMM_H__

#include <inttypes.h>
#include <stdio.h>

#include <cstring>
#include <type_traits>

#define FP_DECLARATION_TEMPLATE \
  template <class T,            \
            std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>

void gemm_cpu_fp32(float *OUT, float *A, float *B, float *C, const int64_t i,
                   const int64_t j, const int64_t k);

/*
  OUT = A x B + C

  OUT : (i, k)
  A : (i, j)
  B : (j, k)
  C : (1, k) - will be broadcasted
*/
FP_DECLARATION_TEMPLATE
void gemm_cpu(T *OUT, T *A, T *B, T *C, const int64_t i, const int64_t j,
              const int64_t k) {
  memset(OUT, 0, i * k * sizeof(T));

  if constexpr (std::is_same<T, float>::value) {
    gemm_cpu_fp32(OUT, A, B, C, i, j, k);
  }
  else if (sizeof(T) == 2) {
    printf("fp16 not implemented\n");
  }
  else {
    printf("not implemented type\n");
  }
}

#if defined(__riscv) || defined(__riscv__)
FP_DECLARATION_TEMPLATE
inline void gemm_hwacha(T *OUT, T *A, T *B, T *C, const int64_t i,
                        const int64_t j, const int64_t k);
#endif

#endif
