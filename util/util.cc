/**
 * util.cc
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

#include "util.hh"

void print_vec1d(std::string name, elem_t *matrix, int isize) {
  // printf cannot directly display the value of fp16
  // so we convert fp16 to fp32
  float matrix_fp32[isize];
  f16MatTof32(&matrix_fp32[0], (ushort *)matrix, isize);

  std::cout << name + ':' << std::endl;
  for (int i = 0; i < isize; i++) {
    printf("%2.2f,", matrix_fp32[i]);
  }

  printf("\b\n\n");
}

void print_mat2d(std::string name, elem_t *matrix, int isize, int jsize) {
  // printf cannot directly display the value of fp16
  // so we convert fp16 to fp32
  float matrix_fp32[isize][jsize];
  f16MatTof32(&matrix_fp32[0][0], (ushort *)matrix, isize * jsize);

  std::cout << name + ':' << std::endl;
  for (int i = 0; i < isize; i++) {
    printf("[");
    for (int j = 0; j < jsize; j++) {
      printf("%2.2f,", matrix_fp32[i][j]);
    }
    printf("\b],\n");
  }

  printf("\b\n\n");
}

void print_mat3d(std::string name, elem_t *matrix, int isize, int jsize,
                 int ksize) {
  // printf cannot directly display the value of fp16
  // so we convert fp16 to fp32
  float matrix_fp32[isize][jsize][ksize];
  f16MatTof32(&matrix_fp32[0][0][0], (ushort *)matrix, isize * jsize * ksize);

  std::cout << name + ':' << std::endl;
  for (int i = 0; i < isize; i++) {
    printf("[");
    for (int j = 0; j < jsize; j++) {
      printf("[");
      for (int k = 0; k < ksize; k++) {
        printf("%2.2f,", matrix_fp32[i][j][k]);
      }
      printf("\b],");
    }
    printf("\b],\n");
  }

  printf("\b\n\n");
}

bool is_vec_equal(elem_t *a, elem_t *b, int len) {
  for (int i = 0; i < len; i++)
    if (a[i] != b[i])
      return false;
  return true;
}

bool is_vec_equal(float *a, float *b, int len) {
  for (int i = 0; i < len; i++)
    if (a[i] != b[i])
      return false;
  return true;
}

bool is_vec_similar(float *a, float *b, int len) {
  for (int i = 0; i < len; i++) {
    float diff = a[i] - b[i];

    if (-0.0001f < diff && diff < 0.0001f)
      return false;
  }
  return true;
}
