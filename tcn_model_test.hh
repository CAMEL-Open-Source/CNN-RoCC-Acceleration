/**
 * tcn_model_test.hh
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
#include "modules/tcn_model.hh"
#include "modules/layer.hh"
#include "modules/tcnblock.hh"
#include "util/bfloat16.h"
#include "util/util.hh"

#if DEBUG
#define BATCH_SIZE 1
#define IN_FEATURES 3
#define OUT_CATEGORIES 2
#define IN_LENGTH 8
#define KERNEL_SIZE 3
#define MATMUL_TYPE OS
#else
#define BATCH_SIZE 1
#define IN_FEATURES 7
#define OUT_CATEGORIES 6
#define IN_LENGTH 16
#define KERNEL_SIZE 3
#define MATMUL_TYPE OS
#endif

#ifndef NUM_BLOCKS
#error "# Layers not set in compiler"
#endif

static float input_fp32[BATCH_SIZE][IN_LENGTH][IN_FEATURES] = {{{0., 1., 0.},
                                                                {3., 3., 0.},
                                                                {1., 1., 2.},
                                                                {0., 2., 1.},
                                                                {3., 0., 2.},
                                                                {3., 3., 3.},
                                                                {3., 2., 3.},
                                                                {3., 0., 2.}}};

static float weight1_fp32[HIDDEN_FEATURES][KERNEL_SIZE][IN_FEATURES] = {
    {{1., 1., 0.}, {1., 2., 1.}, {2., 2., 0.}},

    {{1., 0., 2.}, {0., 1., 1.}, {0., 0., 1.}},

    {{0., 1., 2.}, {1., 2., 2.}, {0., 1., 2.}},

    {{2., 2., 0.}, {1., 2., 0.}, {2., 2., 0.}}};
static float bias1_fp32[HIDDEN_FEATURES] = {1., 1., 0., 1.};

static float weight2_fp32[HIDDEN_FEATURES][KERNEL_SIZE][HIDDEN_FEATURES] = {
    {{2., 0., 1., 0.}, {0., 1., 1., 0.}, {0., 2., 0., 0.}},

    {{0., 2., 2., 0.}, {2., 2., 2., 2.}, {2., 0., 0., 1.}},

    {{2., 2., 0., 2.}, {1., 1., 1., 2.}, {0., 2., 0., 2.}},

    {{2., 2., 0., 0.}, {0., 1., 0., 0.}, {2., 0., 2., 2.}}};
static float bias2_fp32[HIDDEN_FEATURES] = {0., 2., 2., 0.};

static float weight1x1_fp32[HIDDEN_FEATURES][1][IN_FEATURES] = {{{1., 1., 2.}},

                                                                {{0., 1., 1.}},

                                                                {{0., 0., 1.}},

                                                                {{1., 0., 0.}}};
static float bias1x1_fp32[HIDDEN_FEATURES] = {0., 1., 2., 1.};

static float weight_linear_fp32[HIDDEN_FEATURES][OUT_CATEGORIES] = {
    {0., 1.}, {0., 1.}, {1., 1.}, {0., 0.}};
static float bias_linear_fp32[OUT_CATEGORIES] = {2., 2.};

static float output_fp32[BATCH_SIZE][OUT_CATEGORIES] = {{312., 782.}};

// {{{  3.,  13.,  12.,  15.},
//   { 12.,  67.,  49.,  75.},
//   { 32., 130., 116., 104.},
//   { 76., 176., 193., 141.},
//   { 96., 206., 210., 142.},
//   {100., 223., 221., 183.},
//   {116., 274., 261., 220.},
//   {145., 325., 310., 227.}}}

// fp16
static elem_t input[BATCH_SIZE][IN_LENGTH][IN_FEATURES];

static elem_t weight1[HIDDEN_FEATURES][KERNEL_SIZE][IN_FEATURES];
static acc_t bias1[HIDDEN_FEATURES];

static elem_t weight2[HIDDEN_FEATURES][KERNEL_SIZE][HIDDEN_FEATURES];
static acc_t bias2[HIDDEN_FEATURES];

static elem_t weight1x1[HIDDEN_FEATURES][1][IN_FEATURES];
static acc_t bias1x1[HIDDEN_FEATURES];

static elem_t weight_linear[HIDDEN_FEATURES][OUT_CATEGORIES];
static acc_t bias_linear[OUT_CATEGORIES];

static elem_t output[BATCH_SIZE][OUT_CATEGORIES];

static elem_t output_mat[BATCH_SIZE][OUT_CATEGORIES];
