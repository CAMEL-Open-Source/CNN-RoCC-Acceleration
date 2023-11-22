#include "tcnblock.hh"

BaseTCNBlock::BaseTCNBlock(int batch_size, int in_features, int out_features,
                           int in_length)
    : Layer(batch_size, in_features, out_features) {
  length = in_length;
  hidden_size = batch_size * length * out_features;

  //  else if (mode == DWS) {
  //   conv1 = new DWSConv1d(batch_size, in_features, out_features, in_length,
  //                         kernel_size, stride, dilation.first, padding1,
  //                         activation, has_bias);
  //   conv2 = new DWSConv1d(batch_size, out_features, out_features, in_length,
  //                         kernel_size, stride, dilation.second, padding2,
  //                         activation, has_bias);
  //   conv1x1 = (in_features != out_features)
  //                 ? new Conv1d(batch_size, in_features, out_features,
  //                 in_length,
  //                              1, 1, 1, 0, NO_ACTIVATION)
  //                 : nullptr;
  // }
}

BaseTCNBlock::~BaseTCNBlock() {
  delete conv1;
  delete conv2;
  delete conv1x1;

  free(conv1_out);
  free(conv2_out);
  free(conv1x1_out);
}
BaseTCNBlock::BaseTCNBlock(BaseTCNBlock &&o)
    : Layer(o.batch_size, o.in_features, o.out_features) {
  this->conv1 = std::move(o.conv1);
  this->conv2 = std::move(o.conv2);
  this->conv1x1 = std::move(o.conv1x1);
  this->length = std::move(o.length);
  this->res_shift = std::move(o.res_shift);
  this->conv1_out = std::move(o.conv1_out);
  this->conv2_out = std::move(o.conv2_out);
  this->conv1x1 = std::move(o.conv1x1);

  o.conv1 = nullptr;
  o.conv2 = nullptr;
  o.conv1x1 = nullptr;
  o.length = 0;
  o.res_shift = 0;
  o.conv1_out = nullptr;
  o.conv2_out = nullptr;
  o.conv1x1_out = nullptr;
}

int BaseTCNBlock::prepare() {
  conv1->prepare();
  conv2->prepare();
  if (conv1x1)
    conv1x1->prepare();

  conv1_out = (elem_t *)malloc(sizeof(elem_t) * hidden_size);
  conv2_out = (elem_t *)malloc(sizeof(elem_t) * hidden_size);
  conv1x1_out =
      conv1x1 ? (elem_t *)malloc(sizeof(elem_t) * hidden_size) : nullptr;

  return 0;
}

int BaseTCNBlock::prepare(std::vector<elem_t *> &params,
                          std::vector<int> &scales) {
  res_shift = scales[0];

  auto param_begin = params.begin();
  std::vector<elem_t *> conv1_param(param_begin + CONV1_PARAM_START_IDX,
                                    param_begin + CONV2_PARAM_START_IDX);
  std::vector<elem_t *> conv2_param(param_begin + CONV2_PARAM_START_IDX,
                                    param_begin + CONV1X1_PARAM_START_IDX);
  std::vector<elem_t *> conv1x1_param(param_begin + CONV1X1_PARAM_START_IDX,
                                      params.end());

  auto scale_begin = scales.begin();
  std::vector<int> conv1_scale(scale_begin + CONV1_SCALE_START_IDX,
                               scale_begin + CONV2_SCALE_START_IDX);
  std::vector<int> conv2_scale(scale_begin + CONV2_SCALE_START_IDX,
                               scale_begin + CONV1X1_SCALE_START_IDX);
  std::vector<int> conv1x1_scale(scale_begin + CONV1X1_SCALE_START_IDX,
                                 scales.end());

  conv1->prepare(conv1_param, conv1_scale);
  conv2->prepare(conv2_param, conv2_scale);
  if (conv1x1)
    conv1x1->prepare(conv1x1_param, conv1x1_scale);

  conv1_out = (elem_t *)malloc(sizeof(elem_t) * hidden_size);
  conv2_out = (elem_t *)malloc(sizeof(elem_t) * hidden_size);
  conv1x1_out =
      conv1x1 ? (elem_t *)malloc(sizeof(elem_t) * hidden_size) : nullptr;

  return 0;
}

int BaseTCNBlock::forward(elem_t *input, elem_t *output) {
#if DEBUG
  printf("base tcnblock\n");
#endif
  conv1->forward(input, conv1_out);
  conv2->forward(conv1_out, conv2_out);

  if (conv1x1) {
    conv1x1->forward(input, conv1x1_out);
    tiled_resadd_auto(batch_size * length, out_features, 0, conv1x1_out,
                      conv2_out, output, true, WS);
  } else {
    tiled_resadd_auto(batch_size * length, out_features, 0, input, conv2_out,
                      output, true, WS);
  }
  return 0;
}

void BaseTCNBlock::print_weights() {
  printf("TCN Block\n");
  conv1->print_weights();
  conv2->print_weights();
  if (conv1x1)
    conv1x1->print_weights();
}

OriginalTCNBlock::OriginalTCNBlock(int batch_size, int in_features,
                                   int out_features, int in_length,
                                   int kernel_size, int stride,
                                   std::pair<int, int> dilation, int activation,
                                   bool has_bias,
                                   tiled_matmul_type_t matmul_type)
    : BaseTCNBlock(batch_size, in_features, out_features, in_length) {
  int padding1 = (kernel_size - 1) * dilation.first;
  int padding2 = (kernel_size - 1) * dilation.second;

  conv1 = new Conv1d(batch_size, in_features, out_features, in_length,
                     kernel_size, stride, dilation.first, padding1, activation,
                     has_bias, matmul_type);
  conv2 = new Conv1d(batch_size, out_features, out_features, in_length,
                     kernel_size, stride, dilation.second, padding2, activation,
                     has_bias, matmul_type);
  conv1x1 = (in_features != out_features)
                ? new Conv1d(batch_size, in_features, out_features, in_length,
                             1, 1, 1, 0, NO_ACTIVATION, has_bias, matmul_type)
                : nullptr;
}

StatefulTCNBlock::StatefulTCNBlock(int batch_size, int in_features,
                                   int out_features, int in_length,
                                   int kernel_size, int stride,
                                   std::pair<int, int> dilation, int activation,
                                   bool has_bias, tiled_matmul_type_t matmul_type)
    : BaseTCNBlock(batch_size, in_features, out_features, in_length) {

  int padding1 = (kernel_size - 1) * dilation.first;
  int padding2 = (kernel_size - 1) * dilation.second;

  conv1 = new StatefulConv1d(batch_size, in_features, out_features, in_length,
                             kernel_size, stride, dilation.first, padding1,
                             activation, has_bias, matmul_type);
  conv2 = new StatefulConv1d(batch_size, out_features, out_features, in_length,
                             kernel_size, stride, dilation.second, padding2,
                             activation, has_bias, matmul_type);
  conv1x1 = (in_features != out_features)
                ? new StatefulConv1d(batch_size, in_features, out_features,
                                     in_length, 1, 1, 1, 0, NO_ACTIVATION, has_bias, matmul_type)
                : nullptr;

  hidden_size = batch_size * out_features;
}

int StatefulTCNBlock::forward(elem_t *input, elem_t *output) {
#if DEBUG
  printf("stateful tcnblock\n");
#endif
  conv1->forward(input, conv1_out);
  conv2->forward(conv1_out, conv2_out);

  if (conv1x1) {
    conv1x1->forward(input, conv1x1_out);
    tiled_resadd_auto(batch_size, out_features, 0, conv1x1_out, conv2_out,
                      output, true, WS);
  } else {
    tiled_resadd_auto(batch_size, out_features, 0, input, conv2_out, output,
                      true, WS);
  }
  return 0;
}
