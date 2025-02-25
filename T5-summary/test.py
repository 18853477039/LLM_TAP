import os

from datasets import load_dataset

data_path = '/root/autodl-tmp/data/summary/'
path = os.path.join(data_path, 'xsum.py')
cache_dir = os.path.join(data_path, 'cache')
data_files = {'data': data_path}

dataset = load_dataset(path, data_files=data_files, cache_dir=cache_dir)

print(dataset)