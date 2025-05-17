/**
 * util.hh
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

// check if two vectors have exactly same value
bool is_vec_equal(elem_t *a, elem_t *b, int len);
bool is_vec_equal(float *a, float *b, int len);

// check if two vectors have exactly similar value (diff < 0.0001)
bool is_vec_similar(float *a, float *b, int len);

#endif
