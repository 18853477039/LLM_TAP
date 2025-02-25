# 创建一个小模型
from transformers import AutoConfig


config = AutoConfig.for_model('llama')
config.hidden_size = 24
config.intermediate_size = config.hidden_size * 4
config.num_attention_heads = 4
config.num_hidden_heads = 4
config.num_key_value_heads = 2
config.vocab_size = 128

# 实例化模型
from transformers import AutoModel, AutoModelForCausalLM

raw_model = AutoModel.from_config(config)  # 没带因果头