# dummy_dws_conv1d.py
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
import string
torch.manual_seed(0)

# USAGE : python dummy_dws_conv1d.py

BATCH_SIZE = 1
IN_LENGTH = 8
IN_CHANNELS = 3
OUT_CHANNELS = 4
KERNEL_DIM = 3
PADDING = 2
STRIDE = 1
DILATION = 1

input = torch.randint(0, 5,
                      (BATCH_SIZE, IN_CHANNELS, IN_LENGTH),
                      dtype=torch.float32)

depthwise = nn.Conv1d(IN_CHANNELS, IN_CHANNELS, KERNEL_DIM,
                      STRIDE, PADDING, DILATION,
                      groups=IN_CHANNELS, bias=True)
pointwise = nn.Conv1d(IN_CHANNELS, OUT_CHANNELS, kernel_size=1,
                      stride=1, padding=0, dilation=1,
                      bias=True)

depthwise.weight.data.copy_(
    torch.randint_like(depthwise.weight.data, 0, 5, dtype=torch.float32))
depthwise.bias.data.copy_(
    torch.randint_like(depthwise.bias.data, 0, 5, dtype=torch.float32))
pointwise.weight.data.copy_(
    torch.randint_like(pointwise.weight.data, 0, 5, dtype=torch.float32))
pointwise.bias.data.copy_(
    torch.randint_like(pointwise.bias.data, 0, 5, dtype=torch.float32))


with torch.no_grad():
    hidden = depthwise(input)
    output = pointwise(hidden)
    output = output[:, :, :-PADDING]

bracket_to_curly = str.maketrans("[]", "{}")
print('input')
print(str(input.transpose(1, 2)).translate(bracket_to_curly))
print('dw_weight')
print(str(depthwise.weight.transpose(1, 2)).translate(bracket_to_curly))
print('dw_bias')
print(str(depthwise.bias).translate(bracket_to_curly))
print('pw_weight')
print(str(pointwise.weight.transpose(1, 2)).translate(bracket_to_curly))
print('pw_bias')
print(str(pointwise.bias).translate(bracket_to_curly))

print('hidden')
print(str(hidden.transpose(1, 2)).translate(bracket_to_curly))
print('output')
print(str(output.transpose(1, 2)).translate(bracket_to_curly))
