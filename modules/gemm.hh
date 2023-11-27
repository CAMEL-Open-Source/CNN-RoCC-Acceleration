// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#ifndef __SRC_LAYERS_GEMM_H__
#define __SRC_LAYERS_GEMM_H__
void gemm_cpu(float *OUT, float *A, float *B, float *C, const int64_t i, const int64_t j,
              const int64_t k);
#endif
