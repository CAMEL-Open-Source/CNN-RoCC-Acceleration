// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#ifndef __UTIL_HH__
#define __UTIL_HH__

#include <iostream>
#include <string>

#include "bfloat16.h"
#include "gemmini-api/gemmini_params.h"

#define ARRAY_INDEX_2D(jsize, i, j) ((i) * (jsize) + (j))
#define ARRAY_INDEX_3D(jsize, ksize, i, j, k) \
  ((i) * (jsize * ksize) + (j) * (ksize) + (k))

void print_vec1d(std::string name, elem_t *matrix, int isize);
void print_mat2d(std::string name, elem_t *matrix, int isize, int jsize);
void print_mat3d(std::string name, elem_t *matrix, int isize, int jsize,
                 int ksize);

bool is_vec_equal(elem_t *a, elem_t *b, int len);
bool is_vec_equal(float *a, float *b, int len);
bool is_vec_similar(float *a, float *b, int len);

#endif
