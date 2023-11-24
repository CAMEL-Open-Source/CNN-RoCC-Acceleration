// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#ifndef __GEMMINI_API_GEMMINI_PARAMS_H__
#define __GEMMINI_API_GEMMINI_PARAMS_H__

#include <limits.h>
#include <stdint.h>

#if MODE_VC707

#define DIM 8
#define ADDR_LEN 32
#define BANK_NUM 4
#define BANK_ROWS 4096
#define ACC_ROWS 4096
#define MAX_BYTES 64
#define MAX_BLOCK_LEN (MAX_BYTES / (DIM * 2))
#define MAX_BLOCK_LEN_ACC (MAX_BYTES / (DIM * 2))

#elif MODE_VCU118

#define DIM 16
#define ADDR_LEN 32
#define BANK_NUM 4
#define BANK_ROWS 2048
#define ACC_ROWS 2048
#define MAX_BYTES 64
#define MAX_BLOCK_LEN (MAX_BYTES/(DIM*2))
#define MAX_BLOCK_LEN_ACC (MAX_BYTES/(DIM*2))

#else
#error "Unexpected FPGA type"
#endif

typedef uint16_t elem_t;
#define elem_t_max 0x7f7f
#define elem_t_min 0xff7f
typedef uint16_t acc_t;
typedef double full_t;

#define ELEM_T_IS_FLOAT
#define ELEM_T_EXP_BITS 5
#define ELEM_T_SIG_BITS 11
#define ACC_T_EXP_BITS 5
#define ACC_T_SIG_BITS 11
typedef uint16_t elem_t_bits;
typedef uint16_t acc_t_bits;

typedef int32_t scale_t;
typedef uint32_t scale_t_bits;

typedef int32_t scale_acc_t;
typedef uint32_t scale_acc_t_bits;

#define row_align(blocks)                                                      \
  __attribute__((aligned(blocks * DIM * sizeof(elem_t))))
#define row_align_acc(blocks)                                                  \
  __attribute__((aligned(blocks * DIM * sizeof(acc_t))))

#define MVIN_SCALE_ONE 1

#endif // GEMMINI_PARAMS_H
