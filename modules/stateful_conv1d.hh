#ifndef __STATEFUL_CONV1D_HH__
#define __STATEFUL_CONV1D_HH__

#include "gemmini-api/gemmini_testutils.h"
#include "layer.hh"

#define STATEFUL_CONV1D_NUM_PARAMS 2
#define STATEFUL_CONV1D_NUM_SCALES 1
class StatefulConv1d : public Conv1d {
  /* bit different from typical conv1d since this applies padding only on left
   * side rather than both side*/
private:
  elem_t *old_in;    // [batch_size][kernel_reach-1][in_features]
  elem_t *in_merged; // [batch_size][kernel_reach][in_features]

public:
  StatefulConv1d(int batch_size, int in_features, int out_features,
                 int in_length, int kernel_size, int stride, int dilation,
                 int padding, int activation = RELU, bool has_bias = true,
                 tiled_matmul_type_t matmul_type = WS);
  ~StatefulConv1d();
  int prepare() override;
  int prepare(std::vector<elem_t *> &, std::vector<int> &) override;
  int forward(elem_t *input,
              elem_t *output) override; // input : [batch_size][in_features]
  int seq2col(const elem_t *input) override;
  int merge_and_save(elem_t *);
  int warmup(elem_t *input, elem_t *output);
};

#endif
