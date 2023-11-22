import torch
import torch.nn as nn
import torch.nn.functional as F
import string

# USAGE : python dummy_conv1d.py

BATCH_SIZE = 1
IN_LENGTH = 8
IN_FEATURES = 3
OUT_FEATURES = 4
KERNEL_DIM = 3
STRIDE = 1
DILATION1 = 1
DILATION2 = 1
groups = 1

torch.manual_seed(0)

padding1 = (KERNEL_DIM-1)*DILATION1
padding2 = (KERNEL_DIM-1)*DILATION2

# initialization
input = torch.randint(0, 5, (BATCH_SIZE, IN_FEATURES, IN_LENGTH))
weight1 = torch.randint(
    0, 5, (OUT_FEATURES, int(IN_FEATURES/groups), KERNEL_DIM))
bias1 = torch.randint(0, 5, (OUT_FEATURES,))
weight2 = torch.randint(
    0, 5, (OUT_FEATURES, int(OUT_FEATURES/groups), KERNEL_DIM))
bias2 = torch.randint(0, 5, (OUT_FEATURES,))
if IN_FEATURES != OUT_FEATURES:
    weight1x1 = torch.randint(
        0, 5, (OUT_FEATURES, int(IN_FEATURES/groups), 1))
    bias1x1 = torch.randint(0, 5, (OUT_FEATURES,))

# computation
output1 = F.conv1d(
    input, weight1, bias1, STRIDE, padding1, DILATION1, groups)
output1 = output1[:, :, :-padding1]
output1 = F.relu(output1)

output2 = F.conv1d(
    output1, weight2, bias2, STRIDE, padding2, DILATION2, groups)
output2 = output2[:, :, :-padding2]
output2 = F.relu(output2)

res = input if IN_FEATURES == OUT_FEATURES else \
    F.conv1d(input, weight1x1, bias1x1, 1)

output = F.relu(res+output2)

# output
bracket_to_curly = str.maketrans("[]", "{}")
print('input')
print(str(input.transpose(1, 2)).translate(bracket_to_curly))
print('weight1')
print(str(weight1.transpose(1, 2)).translate(bracket_to_curly))
print('bias1')
print(str(bias1).translate(bracket_to_curly))
print('weight2')
print(str(weight2.transpose(1, 2)).translate(bracket_to_curly))
print('bias2')
print(str(bias2).translate(bracket_to_curly))
print('weight1x1')
print(str(weight1x1.transpose(1, 2)).translate(bracket_to_curly))
print('bias1x1')
print(str(bias1x1).translate(bracket_to_curly))


print('output1')
print(str(output1.transpose(1, 2)).translate(bracket_to_curly))
print('output2')
print(str(output2.transpose(1, 2)).translate(bracket_to_curly))
print('res')
print(str(res.transpose(1, 2)).translate(bracket_to_curly))

print('output')
print(str(output.transpose(1, 2)).translate(bracket_to_curly))
