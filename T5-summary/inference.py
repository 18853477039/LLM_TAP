# !/usr/bin/env python3
"""
测试训练好的模型效果。
"""


from rich import print
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = 'cuda:0'
max_source_seq_len = 256
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/checkpoints/summary/model_best/')
model = AutoModelForSeq2SeqLM.from_pretrained('/root/autodl-tmp/checkpoints/summary/model_best/')
model.to(device).eval()


def inference(qustion: str, context: str):
    """
    inference函数。

    Args:
        qustion (str): 问题
        context (str): 原文
    """
    input_seq = f'问题：{question}{tokenizer.sep_token}原文：{context}'
    inputs = tokenizer(
        text=input_seq,
        truncation=True,
        max_length=max_source_seq_len,
        padding='max_length',
        return_tensors='pt'
    )
    outputs = model.generate(input_ids=inputs["input_ids"].to(device))
    print(outputs)
    print(outputs[0].cpu().numpy())
    output = tokenizer.decode(outputs[0].cpu().numpy(), skip_special_tokens=True).replace(" ", "")
    print(f'Q: "{qustion}"')
    print(f'C: "{context}"')
    print(f'A: "{output}"')


if __name__ == '__main__':
    # question = '治疗宫颈糜烂的最佳时间'
    # context = '专家指出，宫颈糜烂治疗时间应选在月经干净后3-7日，因为治疗之后宫颈有一定的创面，如赶上月经期易发生感染。因此患者应在月经干净后3天尽快来医院治疗。同时应该注意，术前3天禁同房，有生殖道急性炎症者应治好后才可进行。'
    # question = '小米汽车什么时候上市'
    # context = '小米汽车是中国自主研发的一款电动汽车，于2017年11月29日在上海上市'
    dev_data = {
        "context": "一亩大约等于666.67平方米45亩的话就是666.67乘45等于30000.15平方米，再用30000.15除以100等于300.15 ，建房子的话要有空隙的啊，所以至少可以建 240———250套吧！！！",
        "answer": "30000.15平方米", "question": "45亩等于多少平方米", "id": 79}
    dev_data = {
        "context": "你好,这两年我都有参加一建的考试,所以对报名时间很关注。 15年开始报名时间很早,7月前后各省都开始报名了,我在广东报考6月22号就开始报名了。 16年报名时间很晚,各省基本到7月底才出报名通知,广东更是到8月3号才开始报名。 按这个趋势17年考试报名时间应该也会在7月开始。你可以从六月底关注本身的人事考试网站,一般报名窗口开放半个月时间,这样就不用担心错过报名了。 希望我的回答能帮到你。|2017年一级建造师考试报名时间已经公布,报名时间为6月16日至7月16日。详情登陆中国人事考试网查看。",
        "answer": "7月", "question": "2017一建报名时间", "id": 532}

    question = dev_data['question']
    context = dev_data['context']
    inference(qustion=question, context=context)