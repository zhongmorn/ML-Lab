import torch

from run.configs import CHECKPOINT, CPU_THREADS
from run.predict import evaluate, load_model
from run.train import train

# 切换训练或加载权重预测；QUESTION 为 None 时读取 data/eval.txt。
PREDICT_ONLY = False
QUESTION = None


if __name__ == "__main__":
    # 预测沿用检查点中的模型和词表，并展示生成词元与停止状态。
    if PREDICT_ONLY:
        torch.set_num_threads(CPU_THREADS)
        model, vocab = load_model(CHECKPOINT)
        questions = None
        if QUESTION is not None:
            questions = [QUESTION]
        for result in evaluate(model, vocab, questions):
            print(f"问题：{result['question']}\n预测：{result['prediction']}")
            tokens = [item["token"] for item in result["trace"]]
            print("预测词元：" + " → ".join(tokens))
            print(f"遇到结束标记：{result['stopped_at_eos']}")
    # 训练后保存权重、评估结果和图表。
    else:
        train(CHECKPOINT)
