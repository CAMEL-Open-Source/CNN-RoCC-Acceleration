#ifndef __DWS_CONV1D_HH__
#define __DWS_CONV1D_HH__

#include "layer.hh"
#include "util/util.hh"
#include <array>

#define DEPTHWISECONV1D_NUM_PARAMS 4
#define DEPTHWISECONV1D_NUM_SCALES 2
class DepthwiseConv1d : public Conv1d {
private:
  int groups;

public:
  DepthwiseConv1d(int batch_size, int in_features, int out_features,
                  int in_length, int kernel_size, int stride, int dilation,
                  int padding, int activation = RELU, bool has_bias = true);
  ~DepthwiseConv1d();
  int prepare() override;
  int prepare(std::vector<elem_t *> &, std::vector<int> &) override;
  int seq2col(const elem_t *input) override;
  int forward(elem_t *input, elem_t *output) override;
  void print_weights() override;
};

#define POINTWISECONV1D_NUM_PARAMS 4
#define POINTWISECONV1D_NUM_SCALES 2
class PointwiseConv1d : public Conv1d {
private:
public:
  PointwiseConv1d(int batch_size, int in_features, int out_features,
                  int in_length, int activation = RELU, bool has_bias = true);
  ~PointwiseConv1d() {}
  int seq2col(const elem_t *input) override;
};

#define DWSCONV1D_NUM_PARAMS 4
#define DWSCONV1D_NUM_SCALES 2
class DWSConv1d : public Layer {
private:
  std::array<Conv1d *, 2> convs;
  elem_t *hidden;

  int in_length;
  int out_length; // length of the output sequence

  int kernel_size;
  int stride;
  int dilation;
  int padding;
  int output_scale;
  int activation;
  bool has_bias;

public:
  DWSConv1d(int batch_size, int in_features, int out_features, int in_length,
            int kernel_size, int stride, int dilation, int padding,
            int activation = RELU, bool has_bias = true);
  ~DWSConv1d();
  int prepare() override;
  int prepare(std::vector<elem_t *> &, std::vector<int> &) override;
  int forward(elem_t *input, elem_t *output) override;
  void print_weights() override;
};

#endif
