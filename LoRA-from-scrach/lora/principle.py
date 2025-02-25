# 原理部分
import torch.nn as nn

in_features = 10
out_features = 10
r = 1
x = 1
# 任意定义一个线性层
W = nn.Linear(in_features, out_features, bias=False)
# 两个低秩矩阵
lora_A = nn.Linear(in_features, r, bias=False)
lora_B = nn.Linear(r, out_features, bias=False)
# 原来的前向 Wx
W(x)
# 新的前向： Wx + BAx
W(x) + lora_B(lora_A(x))
