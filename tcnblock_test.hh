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
#include "modules/tcnblock.hh"
#include "util/util.hh"

#define BATCH_SIZE 1
#define IN_LENGTH 8
#define IN_FEATURES 3
#define OUT_FEATURES 4
#define KERNEL_SIZE 3
#define STRIDE 1
#define DILATION1 1
#define DILATION2 1

#define NO_BIAS false
#define DEPTHWISE false

#define PATCH_SIZE (KERNEL_SIZE * IN_FEATURES)
#define N_PATCHES (BATCH_SIZE * IN_LENGTH)

bool is_vec_equal(elem_t *a, elem_t *b, int len) {
  for (int i = 0; i < len; i++)
    if (a[i] != b[i])
      return false;
  return true;
}

static float input_fp32[BATCH_SIZE][IN_LENGTH][IN_FEATURES] = {{{4, 2, 1},
                                                                {4, 3, 3},
                                                                {3, 1, 4},
                                                                {0, 1, 3},
                                                                {3, 1, 1},
                                                                {4, 4, 4},
                                                                {2, 3, 1},
                                                                {3, 1, 4}}};

static float weight1_fp32[OUT_FEATURES][KERNEL_SIZE][IN_FEATURES] = {
    {{4, 4, 1}, {1, 4, 2}, {4, 0, 3}},

    {{0, 2, 3}, {0, 4, 3}, {0, 1, 3}},

    {{1, 2, 2}, {4, 3, 1}, {1, 0, 0}},

    {{4, 1, 1}, {3, 0, 1}, {1, 3, 2}}};
static float bias1_fp32[OUT_FEATURES] = {4, 1, 3, 4};

static float weight2_fp32[OUT_FEATURES][KERNEL_SIZE][OUT_FEATURES] = {
    {{0, 2, 2, 0}, {0, 3, 0, 3}, {3, 2, 0, 1}},

    {{4, 0, 4, 3}, {1, 2, 3, 4}, {1, 4, 4, 3}},

    {{4, 3, 0, 0}, {3, 2, 1, 2}, {0, 3, 1, 2}},

    {{0, 0, 3, 3}, {1, 4, 1, 0}, {1, 0, 2, 4}}};
static float bias2_fp32[OUT_FEATURES] = {4, 3, 4, 4};

static float weight1x1_fp32[OUT_FEATURES][1][IN_FEATURES] = {{{4, 0, 2}},

                                                             {{4, 0, 3}},

                                                             {{3, 2, 3}},

                                                             {{0, 4, 4}}};
static float bias1x1_fp32[OUT_FEATURES] = {3, 1, 3, 1};

static float output_fp32[BATCH_SIZE][IN_LENGTH][OUT_FEATURES] = {
    {{122, 146, 83, 118},
     {308, 516, 328, 330},
     {594, 1096, 693, 634},
     {712, 1435, 979, 833},
     {706, 1473, 1027, 793},
     {639, 1318, 921, 775},
     {669, 1323, 901, 747},
     {807, 1529, 1014, 872}}};

// output1
// tensor({{{23,  6,  7, 16},
//          {43, 24, 30, 36},
//          {75, 42, 44, 52},
//          {59, 42, 38, 48},
//          {49, 32, 25, 32},
//          {48, 35, 31, 42},
//          {60, 40, 44, 47},
//          {80, 49, 44, 49}}})
// output2
// tensor({{{ 101,  126,   61,  105},
//          { 283,  490,  298,  305},
//          { 571, 1071,  667,  613},
//          { 703, 1425,  965,  816},
//          { 689, 1457, 1010,  784},
//          { 612, 1289,  886,  742},
//          { 656, 1311,  883,  730},
//          { 784, 1504,  988,  851}}})

static elem_t input[BATCH_SIZE][IN_LENGTH][IN_FEATURES];
static elem_t weight1[OUT_FEATURES][KERNEL_SIZE][IN_FEATURES];
static acc_t bias1[OUT_FEATURES];
static elem_t weight2[OUT_FEATURES][KERNEL_SIZE][OUT_FEATURES];
static acc_t bias2[OUT_FEATURES];
static elem_t weight1x1[OUT_FEATURES][1][IN_FEATURES];
static acc_t bias1x1[OUT_FEATURES];
static elem_t output[BATCH_SIZE][IN_LENGTH][OUT_FEATURES];

static elem_t output_mat[N_PATCHES][OUT_FEATURES];
