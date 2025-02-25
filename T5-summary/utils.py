# !/usr/bin/env python3
"""
工具类。
"""

import json
import traceback

import numpy as np


def convert_example(examples: dict, tokenizer, max_source_seq_len: int, max_target_seq_len: int):
    """
    Convert examples for tokenization.

    Args:
        examples (dict): Examples from dataset. e.g., -> {
            "document": [...],  # 新闻正文
            "summary": [...],   # 新闻摘要
            "id": [...]         # 唯一标识符
        }
    """
    inputs = []
    targets = []

    for document, summary in zip(examples['document'], examples['summary']):
        # 源文本 (source) 和目标文本 (target)
        source_text = document
        target_text = summary

        # Tokenize source and target
        tokenized_input = tokenizer(
            source_text,
            max_length=max_source_seq_len,
            truncation=True,
            padding='max_length'
        )
        tokenized_target = tokenizer(
            target_text,
            max_length=max_target_seq_len,
            truncation=True,
            padding='max_length'
        )

        inputs.append(tokenized_input)
        targets.append(tokenized_target)

    # Combine input and target
    return {
        "input_ids": [i["input_ids"] for i in inputs],
        "attention_mask": [i["attention_mask"] for i in inputs],
        "labels": [t["input_ids"] for t in targets],
    }


if __name__ == '__main__':
    from rich import print
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("uer/t5-small")

    res = convert_example({
        "document": "Media playback is unsupported on your device\n18 December 2014 Last updated at 10:28 GMT\nMalaysia has successfully tackled poverty over the last four decades by drawing on its rich natural resources.\nAccording to the World Bank, some 49% of Malaysians in 1970 were extremely poor, and that figure has been reduced to 1% today. However, the government's next challenge is to help the lower income group to move up to the middle class, the bank says.\nUlrich Zahau, the World Bank's Southeast Asia director, spoke to the BBC's Jennifer Pak.",
        "summary": "In Malaysia the 'aspirational' low-income part of the population is helping to drive economic growth through consumption, according to the World Bank."
    },
        tokenizer=tokenizer,
        max_source_seq_len=1024,
        max_target_seq_len=128
    )
    print(res)
    print('input_ids: ', tokenizer.convert_ids_to_tokens(res['input_ids'][0]))
    print('labels: ', tokenizer.convert_ids_to_tokens(res['labels'][0]))
