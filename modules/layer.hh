#ifndef __LAYER_H__
#define __LAYER_H__

#include "gemmini-api/gemmini.h"
#include "gemmini-api/gemmini_params.h"
#include "util/util.hh"

#include <cassert>
#include <cstring>
#include <utility>
#include <vector>

// abstract base class
class Layer {
protected:
  int batch_size;
  int in_features;
  int out_features;

public:
  Layer(int, int, int);
  virtual ~Layer() {}

  // memory allocations, define weight & bias
  virtual int prepare() = 0;
  virtual int prepare(std::vector<elem_t *> &, std::vector<int> &) = 0;

  // inference
  virtual int forward(elem_t *, elem_t *) = 0;

  // training
  virtual int backward(elem_t *, elem_t *) { return 0; }

  int get_batch_size() { return batch_size; };
  int get_in_features() { return in_features; };
  int get_out_features() { return out_features; };

  virtual void print_weights() = 0;
};

#define CONV1D_NUM_PARAMS 2
#define CONV1D_NUM_SCALES 1
class Conv1d : public Layer {
  /* bit different from typical conv1d since this applies padding only on left
   * side rather than both side*/
protected:
  int in_length;
  int out_length; // length of the output sequence

  int kernel_size;
  int stride;
  int dilation;
  int padding;
  int output_scale;
  int activation;
  bool has_bias;
  bool depthwise; // mobilenet

  int n_patches;  // number of convolution iterations per output channel
                  // (batchsize*out_length)
  int patch_size; // size of convolution patch (kernel * in_channels)
  int kernel_reach;

  elem_t *seq2col_out;
  elem_t *weight;
  elem_t *bias;

  static constexpr int WEIGHT_IDX = 0;
  static constexpr int BIAS_IDX = 1;

  tiled_matmul_type_t matmul_type;

public:
  Conv1d(int batch_size, int in_features, int out_features, int in_length,
         int kernel_size, int stride, int dilation, int padding,
         int activation = RELU, bool has_bias = true,
         tiled_matmul_type_t matmul_type = WS);
  ~Conv1d();
  int prepare() override;
  int prepare(std::vector<elem_t *> &, std::vector<int> &) override;
  int forward(elem_t *input, elem_t *output) override;
  virtual int seq2col(const elem_t *input);
  void print_weights() override;
  int get_out_length() { return out_length; }
};

#define LINEAR_NUM_PARAMS 2
#define LINEAR_NUM_SCALES 1
class Linear : public Layer {
private:
  int output_scale;
  bool has_bias;
  int activation;
  tiled_matmul_type_t matmul_type;

  elem_t *weight;
  elem_t *bias;

  static constexpr int WEIGHT_IDX = 0;
  static constexpr int BIAS_IDX = 1;

public:
  Linear(int batch_size, int in_features, int out_features, bool has_bias = true,
         int activation = RELU, tiled_matmul_type_t matmul_type = WS);
  ~Linear();

  int prepare() override;
  int prepare(std::vector<elem_t *> &, std::vector<int> &) override;
  int forward(elem_t *input, elem_t *output) override;
  int backward(elem_t *doutput, elem_t *dinput) override;

  void print_weights() override;
};

#endif
