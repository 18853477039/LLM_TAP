import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from PIL.features import modules
from sphinx.ext.autodoc import import_module
from model import raw_model, config
from lora1 import LoraLinear, replace_linear_with_lora, print_trainable_parameters

# 验证lora
## 检查是不是只有lora层是可训练的
def print_model_parameters(model: nn.Module):
    """
    查看模型参数的 requires_grad 情况
    """
    print("Layer Name & Parameters")
    print("------------------------")
    for name, parameter in model.named_parameters():
        print(f"{name:50} | Requires_grad: {parameter.requires_grad}")

# print_model_parameters(lora_model)

## 和hugging face 的 peft对比
validate_peft = False
if validate_peft:
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules='all-linear')
    peft_lora_model = copy.deepcopy(raw_model)
    peft_lora_model = get_peft_model(peft_lora_model, lora_config)
    peft_lora_model.print_trainable_parameters()

## 卸载 LoRA 和 重载 LoRA
# 除了参数，我们还要验证我们的前向是否正确。由于 LoRA 实现特别简单，所以其额外参数部分也很容易卸载。考虑到这个，我们很直接地想到下面四种前向情况：
#
# 原始模型
# LoRA 适配后的模型
# 卸载 LoRA 后的模型
# 重载 LoRA 后的模型
# 如果实现正确，情况一和三的前向结果应该是一致的，情况二和四的前向结果应该是一致的。如果考虑到 BA 不做零初始化，那么这两类前向结果各自应该是不一样的。
#
# 因此，我们前面的 test_mode 字段派上用场，我们对 A 和 B 都做高斯初始化，让 BA 非零，改变模型的前向结果。下面来测试，先写好 unload_lora 和 load_lora。

# unload_lora
from typing import List

def unload_lora(module: nn.Module, adapter_name: str = 'adapter'):
    """
    卸载 LoRA参数，并将原模型恢复到加载LoRA前的样子
    """
    lora_parameters = {}
    def search_lora_linear(module: nn.Module, prefix: str = List[str]):
        for name, child in module.named_children():
            new_prefix = prefix + [name]
            if isinstance(child, LoraLinear):
                lora_parameters['.'.join(new_prefix)] = {
                    "lora_A_weight": child.lora_A.data.cpu(),
                    "lora_B_weight": child.lora_B.data.cpu(),
                    "r": child.r,
                    "alpha": child.alpha,
                    "dropout_p": child.dropout.p,
                }
                setattr(module, name, child.base_layer)
            else:
                search_lora_linear(child, new_prefix)

    search_lora_linear(module, [])

    for name, param in module.named_parameters():
        param.requires_grad = True

    torch.save(lora_parameters, f"{adapter_name}.pt")


# load_lora
def load_lora(module: nn.Module, adapter_name: str = 'adapter'):
    """
    加载 lora参数
    """
    lora_parameters = torch.load(f"{adapter_name}.pt")
    for name, lora_params in lora_parameters.items():
        child = dict(module.named_modules())[name]
        if isinstance(child, nn.Linear):
            lora_linear = LoraLinear(child, r=lora_params['r'], alpha=lora_params['alpha'], dropout_p=lora_params['dropout_p'])
            lora_linear.lora_A.data = lora_params['lora_A_weight'].to(lora_linear.lora_A.device)
            lora_linear.lora_B.data = lora_params['lora_B_weight'].to(lora_linear.lora_B.device)

            # 名称示例 layers.0.self_attn.q_proj
            parts = name.split('.')
            obj = module
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], lora_linear)

    # 恢复原来的冻结方式，这里简单删除了lora全冻结
    for name, param in module.named_parameters():
        if any(s in name for s in ['embed', 'norm', 'lm_head']):
            param.requires_grad = False


# 创建一个测试张量
bsz = 2
seq_len = 8
test_tensor = torch.randint(0, config.vocab_size, (bsz, seq_len))
# 做LoRA替换
##  开测试模式，让BA非零
lora_model = copy.deepcopy(raw_model)
replace_linear_with_lora(lora_model, r=8, alpha=16, test_mode=True)
# 再做四次前向
raw_model.eval()
print_trainable_parameters(raw_model, line_num=109)
raw_res = raw_model(test_tensor).last_hidden_state

# 第一次直接初始化lora的前向结果
lora_model.eval()
print_trainable_parameters(lora_model, line_num=114)
before_unload_res = lora_model(test_tensor).last_hidden_state

# 卸载lora后的前向结果
unload_lora(lora_model)
lora_model.eval()
print_trainable_parameters(lora_model, line_num=120)
unload_res = lora_model(test_tensor).last_hidden_state

# 重新装载lora后的前向结果
load_lora(lora_model)
lora_model.eval()
print_trainable_parameters(lora_model, line_num=126)
load_res = lora_model(test_tensor).last_hidden_state

# 检查前向结果
print('1:', '')
print(torch.allclose(raw_res, unload_res, atol=1e-6))
print('2:', '')
print(torch.allclose(before_unload_res, load_res, atol=1e-6))
print('3:', '')
print(torch.allclose(raw_res, load_res, atol=1e-6))
