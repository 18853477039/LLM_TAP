import copy
import sys

import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL.features import modules
from sphinx.ext.autodoc import import_module


class LoraLinear(nn.Module):
    def __init__(self,
                 base_layer: nn.Linear,    # 原来的线性层
                 r: int = 8,
                 alpha: int = 16,
                 dropout_p: float = 0.0,
                 test_mode: bool = False   # 测试模式，用于控制 lora_B 是否为全零
    ):
        super(LoraLinear, self).__init__()
        self.base_layer = copy.deepcopy(base_layer)
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_p)

        # 定义lora_A 和 lora_B 为 Parameter
        self.lora_A = nn.Parameter(torch.empty((r, base_layer.in_features), dtype=base_layer.weight.dtype))
        self.lora_B = nn.Parameter(torch.empty((base_layer.out_features, r), dtype=base_layer.weight.dtype))

        # 初始化lora矩阵
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        if test_mode:
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.lora_B)

        # 冻结原来的层的参数
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaling = float(self.alpha) / float(self.r)  # lora 缩放系数
        lora_adjustment = F.linear(self.dropout(x), self.lora_A)
        lora_adjustment = F.linear(self.dropout(lora_adjustment), self.lora_B)
        return self.base_layer(x) + lora_adjustment * scaling


def replace_linear_with_lora(
        module: nn.Module,
        r: int = 8,
        alpha: int = 16,
        dropout_p: float = 0.0,
        embed_requires_grad: bool = False, # embedding 层是否训练
        norm_requires_grad: bool = False, # norm 层是否训练
        head_requires_grad: bool = False, # lm_head 层是否训练 (Causal LM才有）
        test_mode: bool = False, # 测试模式，用于控制 lora_B 是否为全零
):
    """
    找到module中所有的线性层
    """
    for name, child in module.named_children():
        # 先处理额外的层，lm_head 也是linear, 所以先处理
        if any(s in name for s in ['embed', 'norm', 'lm_head']):
            requires_grad = embed_requires_grad if 'embed' in name \
                            else norm_requires_grad if 'norm' in name \
                            else head_requires_grad
            for param in child.parameters():
                param.requires_grad = requires_grad

        # 替换所有的线性层，Qlora做法
        elif isinstance(child, nn.Linear):
            lora_linear = LoraLinear(child, r=r, alpha=alpha, dropout_p=dropout_p, test_mode=test_mode)
            setattr(module, name, lora_linear)
        # 递归向下替换
        else:
            replace_linear_with_lora(child, r, alpha, dropout_p, embed_requires_grad, norm_requires_grad, head_requires_grad, test_mode)


def print_trainable_parameters(model: nn.Module, line_num: int =0):
    """
    打印可训练参数，表现和 PeftModel 的 print_trainable_parameters 方法类似
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = trainable_parameters / total_params * 100
    print(f"trainable params: {trainable_parameters: ,} || all params: {total_params:,} || trainable%: {trainable_percent:.4f}, line: {line_num}")

