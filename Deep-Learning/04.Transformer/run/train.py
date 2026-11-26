from pathlib import Path

import torch
from torch.nn import functional as F

from .config import (
    BATCH_SIZE,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    EVAL_DATA,
    LEARNING_RATE,
    LOG_INTERVAL,
    LOSS_AVERAGE_WINDOW,
    MODEL_CONFIG,
    PROJECT_ROOT,
    TRAIN_DATA,
)
from .model import TransformerLM
from .predict import evaluate
from tools.plot import plot_training
from tools.save import save_experiment
from tools.seed import set_seed
from tools.batch import language_model_rows


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

# 只用训练语料建立字符词表，特殊词元在前，其余字符排序以固定编号。
def build_vocab(texts):
    characters = {character for line in texts for character in line}
    return [*SPECIAL_TOKENS, *sorted(characters)]


def load_training_texts():
    texts = [line.strip() for line in TRAIN_DATA.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not texts:
        raise ValueError(f"训练文件没有文本：{TRAIN_DATA}")
    return texts


def train(checkpoint, steps=DEFAULT_STEPS, seed=DEFAULT_SEED):
    if steps < 1:
        raise ValueError("训练步数必须为正整数")
    # 从训练文本构造错开一位的输入/目标窗口，并初始化模型和优化器。
    set_seed(seed)
    corpus = load_training_texts()
    vocab = build_vocab(corpus)
    inputs, targets = language_model_rows(corpus, vocab, MODEL_CONFIG["max_length"])
    model = TransformerLM(len(vocab), **MODEL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # 固定参考批用于比较训练前后的损失；实际更新仍随机抽取训练窗口。
    reference = torch.randint(len(inputs), (BATCH_SIZE,))
    with torch.no_grad():
        initial_loss = F.cross_entropy(model(inputs[reference]).flatten(0, 1), targets[reference].flatten()).item()
    history = []
    # 按批优化下一词交叉熵，并按日志间隔汇总最近若干批的损失。
    for step in range(1, steps + 1):
        batch = torch.randint(len(inputs), (BATCH_SIZE,))
        loss = F.cross_entropy(model(inputs[batch]).flatten(0, 1), targets[batch].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"step": step, "loss": loss.item()})
        if step == 1 or step % LOG_INTERVAL == 0 or step == steps:
            recent = history[-LOSS_AVERAGE_WINDOW:]
            average = sum(item["loss"] for item in recent) / len(recent)
            print(f"步数 {step:4d} | 最近 {len(recent):3d} 批平均损失 {average:.6f}", flush=True)
    # 更新结束后复测参考批，再读取留出问题评估，避免评估内容参与训练。
    model.eval()
    with torch.no_grad():
        final_loss = F.cross_entropy(model(inputs[reference]).flatten(0, 1), targets[reference].flatten()).item()
    evaluations = evaluate(model, vocab)
    # 报告保留训练配置、损失与全部评估结果，第一题同时用于概要和图表。
    report = {
        "seed": seed, "steps": steps, "documents": len(corpus), "training_windows": len(inputs),
        "vocab": vocab, "model_config": MODEL_CONFIG,
        "initial_loss": initial_loss, "final_loss": final_loss, "loss_history": history,
        "train_data": TRAIN_DATA.relative_to(PROJECT_ROOT).as_posix(),
        "eval_data": EVAL_DATA.relative_to(PROJECT_ROOT).as_posix(),
        **evaluations[0],
        "evaluations": evaluations,
    }
    # 保存权重和 JSON 报告，绘制损失及生成候选概率，并输出逐题预测。
    checkpoint = Path(checkpoint)
    save_experiment(checkpoint, model, vocab, report)
    plot_training(report, checkpoint.parent)
    for result in evaluations:
        print(f"留出问题：{result['question']}\n预测：{result['prediction']}\n遇到结束标记：{result['stopped_at_eos']}")
    return report
