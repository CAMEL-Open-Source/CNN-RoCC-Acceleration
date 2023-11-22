import torch
import torch.nn as nn
import string

# USAGE : python dummy_conv1d.py

BATCH_SIZE = 1
IN_LENGTH = 8
IN_CHANNELS = 3
OUT_CHANNELS = 4
KERNEL_DIM = 3
PADDING = 2
STRIDE = 1
DILATION = 1
groups = 1

torch.manual_seed(0)

input = torch.randint(0, 5, (BATCH_SIZE, IN_CHANNELS, IN_LENGTH+2))
weight = torch.randint(
    0, 5, (OUT_CHANNELS, int(IN_CHANNELS/groups), KERNEL_DIM))
bias = torch.randint(0, 5, (OUT_CHANNELS,))

output1 = nn.functional.conv1d(
    input[:, :, :-2], weight, bias, STRIDE, PADDING, DILATION, groups)
output1 = output1[:, :, :-PADDING]
output2 = nn.functional.conv1d(
    input[:, :, :-1], weight, bias, STRIDE, PADDING-1, DILATION, groups)
output2 = output2[:, :, :-PADDING+1]
output3 = nn.functional.conv1d(
    input, weight, bias, STRIDE, 0, DILATION, groups)
output3 = output3[:, :, :]

bracket_to_curly = str.maketrans("[]", "{}")
print('input')
print(str(input.transpose(1, 2)).translate(bracket_to_curly))
print('weight')
print(str(weight.transpose(1, 2)).translate(bracket_to_curly))
print('bias')
print(str(bias).translate(bracket_to_curly))

print('output1')
print(str(output1.transpose(1, 2)).translate(bracket_to_curly))
print('output2')
print(str(output2.transpose(1, 2)).translate(bracket_to_curly))
print('output3')
print(str(output3.transpose(1, 2)).translate(bracket_to_curly))
