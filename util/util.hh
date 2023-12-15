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

static void print_vec1d(std::string name, elem_t *matrix, int isize) {
  float matrix_fp32[isize];
  f16MatTof32(&matrix_fp32[0], (ushort *)matrix, isize);

  std::cout << name + ':' << std::endl;
  for (int i = 0; i < isize; i++) {
    printf("%2.2f,", matrix_fp32[i]);
  }
  printf("\b\n\n");
}

static void print2dmatrix(std::string name, elem_t *matrix, int isize,
                          int jsize) {
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

static void print3dmatrix(std::string name, elem_t *matrix, int isize,
                          int jsize, int ksize) {
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

#endif
