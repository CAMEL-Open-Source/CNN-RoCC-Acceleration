# dummy_interval_time_model.py
# Copyright (C) 2021-2024 CAMELab
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F
import string

# USAGE : python dummy_conv1d.py

BATCH_SIZE = 1
IN_FEATURES = 3
HIDDEN_FEATURES = 4
OUT_CATEGORIES = 2
IN_LENGTH = 8
KERNEL_SIZE = 3
STRIDE = 1
DILATION1 = 1
DILATION2 = 1
groups = 1

torch.manual_seed(0)

# initialization
input = torch.randint(0, 4,
                      (BATCH_SIZE, IN_FEATURES, IN_LENGTH), dtype=torch.float32)


class TemporalBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size,
                 stride, dilation, groups):
        super(TemporalBlock, self).__init__()

        self.padding1 = (kernel_size-1)*dilation[0]
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size,
                               stride=stride,
                               padding=self.padding1,
                               dilation=dilation[0])

        self.padding2 = (kernel_size-1)*dilation[1]
        self.conv2 = nn.Conv1d(out_features, out_features, kernel_size,
                               stride=stride,
                               padding=self.padding2,
                               dilation=dilation[1])

        self.conv1x1 = nn.Conv1d(
            in_features, out_features, 1) if in_features != out_features else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.copy_(
            torch.randint_like(self.conv1.weight.data, 0, 3, dtype=torch.float32))
        self.conv1.bias.data.copy_(
            torch.randint_like(self.conv1.bias.data, 0, 3, dtype=torch.float32))
        self.conv2.weight.data.copy_(
            torch.randint_like(self.conv2.weight.data, 0, 3, dtype=torch.float32))
        self.conv2.bias.data.copy_(
            torch.randint_like(self.conv2.bias.data, 0, 3, dtype=torch.float32))
        self.conv1x1.weight.data.copy_(
            torch.randint_like(self.conv1x1.weight.data, 0, 3, dtype=torch.float32))
        self.conv1x1.bias.data.copy_(
            torch.randint_like(self.conv1x1.bias.data, 0, 3, dtype=torch.float32))

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.padding1]
        out = self.relu(out)

        out = self.conv2(out)
        out = out[:, :, :-self.padding2]
        out = self.relu(out)

        res = x if self.conv1x1 is None else self.conv1x1(x)

        return self.relu(out + res)


tb1 = TemporalBlock(IN_FEATURES, HIDDEN_FEATURES, KERNEL_SIZE,
                    STRIDE, [DILATION1, DILATION2], groups)

linear = nn.Linear(HIDDEN_FEATURES, OUT_CATEGORIES)
linear.weight.data.copy_(
    torch.randint_like(linear.weight.data, 0, 3, dtype=torch.float32))
linear.bias.data.copy_(
    torch.randint_like(linear.bias.data, 0, 3, dtype=torch.float32))


# computation
with torch.no_grad():
    output_tcn = tb1(input)
    output = linear(output_tcn[:, :, -1])

bracket_to_curly = str.maketrans("[]", "{}")


def print_transpose(tensor, dim0, dim1):
    print(str(tensor.transpose(dim0, dim1)).translate(bracket_to_curly))


# output
print('input')
print_transpose(input, 1, 2)

print('weight1')
print_transpose(tb1.conv1.weight.data, 1, 2)
print('bias1')
print(str(tb1.conv1.bias.data).translate(bracket_to_curly))
print('weight2')
print_transpose(tb1.conv2.weight.data, 1, 2)
print('bias2')
print(str(tb1.conv2.bias.data).translate(bracket_to_curly))
print('weight1x1')
print_transpose(tb1.conv1x1.weight.data, 1, 2)
print('bias1x1')
print(str(tb1.conv1x1.bias.data).translate(bracket_to_curly))

print('weight_linear')
print_transpose(linear.weight.data, 0, 1)
print('bias_linear')
print(str(linear.bias.data).translate(bracket_to_curly))

print('tcn')
print_transpose(output_tcn, 1, 2)

print('output')
print(str(output).translate(bracket_to_curly))
