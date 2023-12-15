// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Copyright (C) 2022 CAMELab
 *
 * Author: Junhyeok Jang <jhjang@camelab.org>
 */

#include "util.hh"

bool is_vec_equal(elem_t *a, elem_t *b, int len) {
  for (int i = 0; i < len; i++)
    if (a[i] != b[i])
      return false;
  return true;
}
