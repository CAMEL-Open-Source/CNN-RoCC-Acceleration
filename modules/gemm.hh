// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#ifndef __SRC_LAYERS_GEMM_H__
#define __SRC_LAYERS_GEMM_H__

#include <inttypes.h>
#include <type_traits>

#define FP_DECLARATION_TEMPLATE                                                \
  template <class T,                                                           \
            std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
/*
  OUT = A x B + C

  OUT : (i, k)
  A : (i, j)
  B : (j, k)
  C : (1, k) - will be broadcasted
*/
FP_DECLARATION_TEMPLATE
void gemm_cpu(T *OUT, T *A, T *B, T *C, const int64_t i, const int64_t j,
              const int64_t k);
#endif
