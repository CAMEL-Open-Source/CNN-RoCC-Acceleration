#ifndef __INTERVAL_TIME_MODEL_HH__
#define __INTERVAL_TIME_MODEL_HH__

#include "gemmini-api/gemmini_testutils.h"
#include "layer.hh"
#include "tcnblock.hh"
#include "util/util.hh"
#include <cstring>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/mman.h>
#include <vector>

class IntervaltimeModel {
private:
  int batch_size;
  int in_feature;
  int hidden_feature;
  int out_category;
  int length;

  int num_blocks;
  int kernel_size;
  Mode mode;

  std::vector<Layer *> network; // layers
  std::vector<elem_t *> hidden; // intermediate outputs

  std::vector<uint64_t> ticks;

public:
  IntervaltimeModel(int, int, int, int, int, int, int, Mode = ORIGINAL,
                    tiled_matmul_type_t = WS);
  ~IntervaltimeModel();

  int prepare();
  int prepare(std::vector<elem_t *> &, std::vector<int> &);
  int forward(elem_t *, elem_t *);

  std::vector<uint64_t> &getTicks() { return ticks; }

  void print_weights();
};

#endif
