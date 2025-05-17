# dummy_conv1d.py
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

# USAGE : python dummy_conv1d.py

BATCH_SIZE = 1
IN_LENGTH = 8
IN_CHANNELS = 3
OUT_CHANNELS = 4
KERNEL_DIM = 3
PADDING = 4
STRIDE = 1
DILATION = 2
groups = 1

input = torch.randint(0, 5, (BATCH_SIZE, IN_CHANNELS, IN_LENGTH))
weight = torch.randint(
    0, 5, (OUT_CHANNELS, int(IN_CHANNELS/groups), KERNEL_DIM))
bias = torch.randint(0, 5, (OUT_CHANNELS,))

output = nn.functional.conv1d(
    input, weight, bias, STRIDE, PADDING, DILATION, groups)
output = output[:, :, :-PADDING]

bracket_to_curly = str.maketrans("[]", "{}")
print('input')
print(str(input.transpose(1, 2)).translate(bracket_to_curly))
print('weight')
print(str(weight.transpose(1, 2)).translate(bracket_to_curly))
print('bias')
print(str(bias).translate(bracket_to_curly))
print('output')
print(str(output.transpose(1, 2)).translate(bracket_to_curly))
