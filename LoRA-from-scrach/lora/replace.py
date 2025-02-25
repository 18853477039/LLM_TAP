import copy

from lora1 import print_trainable_parameters, replace_linear_with_lora
from model import raw_model
# raw_model = AutoModelForCausalLM.from_config(config) # 带了因果头
print(raw_model)


# 查看参数情况
print_trainable_parameters(raw_model)

# 替换目标线性层
lora_model = copy.deepcopy(raw_model)
## 替换
replace_linear_with_lora(lora_model, r=8, alpha=16)
## 打印参数情况
print_trainable_parameters(lora_model)

