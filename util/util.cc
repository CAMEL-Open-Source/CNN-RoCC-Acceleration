// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include "util.hh"
void print_vec1d(std::string name, elem_t *matrix, int isize) {
}

void print_mat2d(std::string name, elem_t *matrix, int isize,
                          int jsize) {
}

void print_mat3d(std::string name, elem_t *matrix, int isize,
                          int jsize, int ksize) {
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
