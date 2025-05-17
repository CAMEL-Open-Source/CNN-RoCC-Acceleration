/**
 * dws_conv1d_test.hh
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "gemmini-api/gemmini_testutils.h"
#include "modules/dws_conv1d.hh"
#include "util/util.hh"

// test configuration
#define BATCH_SIZE 1
#define IN_LENGTH 8
#define IN_FEATURES 3
#define OUT_FEATURES 4
#define KERNEL_SIZE 3
#define PADDING 2
#define STRIDE 1
#define DILATION 1

#define BIAS true

static float input_fp32[BATCH_SIZE][IN_LENGTH][IN_FEATURES] = {{{4., 2., 1.},
                                                                {4., 3., 3.},
                                                                {3., 1., 4.},
                                                                {0., 1., 3.},
                                                                {3., 1., 1.},
                                                                {4., 4., 4.},
                                                                {2., 3., 1.},
                                                                {3., 1., 4.}}};

static float dw_weight_fp32[IN_FEATURES][KERNEL_SIZE][1] = {{{3.}, {1.}, {1.}},

                                                            {{0.}, {3.}, {1.}},

                                                            {{1.}, {2.}, {4.}}};
static float dw_bias_fp32[IN_FEATURES] = {1., 3., 4.};

static float pw_weight_fp32[OUT_FEATURES][1][IN_FEATURES] = {{{0., 0., 3.}},

                                                             {{2., 3., 2.}},

                                                             {{2., 0., 0.}},

                                                             {{0., 3., 1.}}};
static float pw_bias_fp32[OUT_FEATURES] = {4., 1., 1., 0.};

static float output_fp32[BATCH_SIZE][IN_LENGTH][OUT_FEATURES] = {
    {{28., 42., 11., 23.},
     {58., 91., 19., 54.},
     {85., 134., 41., 66.},
     {85., 108., 33., 48.},
     {58., 84., 27., 39.},
     {79., 97., 17., 55.},
     {55., 121., 33., 71.},
     {82., 128., 37., 65.}}};

// hidden
// tensor({{{ 5.,  5.,  8.},
//          { 9., 12., 18.},
//          {20., 13., 27.},
//          {16.,  7., 27.},
//          {13.,  7., 18.},
//          { 8., 10., 25.},
//          {16., 18., 17.},
//          {18., 13., 26.},
//          {10.,  6., 13.},
//          {10.,  3.,  8.}}})

static elem_t input[BATCH_SIZE][IN_LENGTH][IN_FEATURES];
static elem_t dw_weight[IN_FEATURES][KERNEL_SIZE][1];
static acc_t dw_bias[IN_FEATURES];
static elem_t pw_weight[OUT_FEATURES][1][IN_FEATURES];
static acc_t pw_bias[OUT_FEATURES];
static elem_t output[BATCH_SIZE][IN_LENGTH][OUT_FEATURES];

static elem_t output_mat[BATCH_SIZE * IN_LENGTH][OUT_FEATURES];
