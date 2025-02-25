# !/usr/bin/env python3
"""
使用T5进行中文问答任务训练，数据集使用百度开源中文问答数据集。
"""


import os
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_scheduler

from utils import convert_example
from iTrainingLogger import iSummaryWriter
from rouge_score import rouge_scorer


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model", default='uer/t5-base-chinese-cluecorpussmall', type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_source_seq_len", default=512, type=int,help="The maximum total encoder input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--max_target_seq_len", default=512, type=int,help="The maximum total decoder input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument("--num_labels", default=2, type=int, help="Total classes of labels.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, data_loader, tokenizer):
    """
    使用 rouge_scorer 在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的 dataloader
    Returns:
        (float, float, float): 返回 ROUGE-1, ROUGE-2, ROUGE-L 的平均 F1 分数
    """
    model.eval()

    # 初始化 ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # 累积分数和计数
    rouge1_f1, rouge2_f1, rougeL_f1 = 0.0, 0.0, 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            # 模型生成预测
            outputs = model.generate(
                input_ids=batch['input_ids'].to(args.device),
                max_length=args.max_target_seq_len
            )

            # 解码预测和参考序列
            decoded_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_references = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

            # 逐一计算 ROUGE 分数
            for prediction, reference in zip(decoded_predictions, decoded_references):
                scores = scorer.score(reference, prediction)
                rouge1_f1 += scores['rouge1'].fmeasure
                rouge2_f1 += scores['rouge2'].fmeasure
                rougeL_f1 += scores['rougeL'].fmeasure
                num_samples += 1

    # 计算平均分数
    rouge1_f1_avg = rouge1_f1 / num_samples if num_samples > 0 else 0.0
    rouge2_f1_avg = rouge2_f1 / num_samples if num_samples > 0 else 0.0
    rougeL_f1_avg = rougeL_f1 / num_samples if num_samples > 0 else 0.0

    model.train()  # 恢复模型到训练模式
    return rouge1_f1_avg, rouge2_f1_avg, rougeL_f1_avg


def train():
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    data_path = '/root/autodl-tmp/data/summary/'
    path = os.path.join(data_path, 'xsum.py')
    cache_dir = os.path.join(data_path, 'cache')
    data_dir = data_path

    dataset = load_dataset(path, data_dir=data_dir, cache_dir=cache_dir)

    print(dataset)
    convert_func = partial(
        convert_example, 
        tokenizer=tokenizer, 
        max_source_seq_len=args.max_source_seq_len,
        max_target_seq_len=args.max_target_seq_len,
    )
    dataset = dataset.map(convert_func, batched=True)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
    model.to(args.device)

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = []
    tic_train = time.time()
    global_step, best_bleu4 = 0, 0
    global_step, best_rougeL = 0, 0
    # saved_checkpoints = []
    # max_checkpoints = 3
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            outputs = model(
                input_ids=batch['input_ids'].to(args.device),
                attention_mask=batch['attention_mask'].to(args.device),
                labels=batch['labels'].to(args.device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                model.save_pretrained(os.path.join(cur_save_dir))
                tokenizer.save_pretrained(os.path.join(cur_save_dir))
                # if len(saved_checkpoints) > max_checkpoints:
                #     oldest_checkpoint = saved_checkpoints.pop(0)  # 移除最旧的路径
                #     if os.path.exists(oldest_checkpoint):
                #         import shutil
                #         shutil.rmtree(oldest_checkpoint)  # 删除最旧的检查点文件夹
                #         print(f"Deleted old checkpoint: {oldest_checkpoint}")

                rouge1, rouge2, rougeL = evaluate_model(model, eval_dataloader, tokenizer)
                writer.add_scalar('eval/rouge-size-1', rouge1, global_step)
                writer.add_scalar('eval/rouge-size-2', rouge2, global_step)
                writer.add_scalar('eval/rouge-size-L', rougeL, global_step)
                writer.record()
                print("Evaluation rougeL: %.5f" % (rougeL))
                
                if rougeL > best_rougeL:
                    print(
                        f"best rougeL performence has been updated: {best_rougeL:.5f} --> {rougeL:.5f}"
                    )
                    best_rougeL = rougeL
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    model.save_pretrained(os.path.join(cur_save_dir))
                    tokenizer.save_pretrained(os.path.join(cur_save_dir))
                tic_train = time.time()


if __name__ == '__main__':
    from rich import print
    train()