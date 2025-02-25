from torch import nn


def attention(query, key, value, mask=None, dropout=None):
    "compute 'scaled dot product attention'"
    d_k = query.zise(-1)



dropout = 0.1
res = nn.Dropout(p=dropout)
print(res)

nn.Linear