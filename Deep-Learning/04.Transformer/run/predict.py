import torch

from tools.batch import encode
from .config import EVAL_DATA, MAX_NEW_TOKENS
from .model import TransformerLM


def load_eval_questions():
    questions = [line.strip() for line in EVAL_DATA.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not questions:
        raise ValueError(f"评估文件没有问题：{EVAL_DATA}")
    return questions


def load_model(checkpoint):
    # 恢复训练时的词表、模型尺寸和参数；缺少尺寸配置时沿用旧问答模型设置。
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved.get("model_config", {"width": 4, "num_heads": 1, "max_length": 32})
    vocab = saved["vocab"]
    state = saved["state_dict"]
    # 旧检查点的可学习位置向量需要单独恢复到当前的位置编码缓冲区。
    legacy_positions = state.pop("position_embedding.weight", None)
    model = TransformerLM(len(vocab), **config)
    if legacy_positions is not None:
        model.position_encoding.copy_(legacy_positions)
    model.load_state_dict(state)
    return model.eval(), vocab


@torch.no_grad()
def predict(model, vocab, question, max_new_tokens=MAX_NEW_TOKENS):
    # 字符模型使用 BOS 前缀，旧问答模型沿用问题后的 SEP 分隔符。
    ids = {token: i for i, token in enumerate(vocab)}
    prompt = encode(question, vocab)
    if "<BOS>" in ids:
        prefix = [ids["<BOS>"], *prompt]
    else:
        prefix = [*prompt, ids["<SEP>"]]
    tokens = torch.tensor([prefix], dtype=torch.long)
    parts = []
    trace = []
    # 自回归生成：沿概率最高的词元继续输入，遇到 EOS 或上下文/生成长度上限停止。
    for _ in range(max_new_tokens):
        if tokens.size(1) > model.position_encoding.size(0):
            break
        # 保留完整词表 softmax 下前两名的原始概率，供逐步预测展示和绘图。
        probabilities = model(tokens)[0, -1].softmax(-1)
        first, second = probabilities.topk(2).indices.tolist()
        trace.append({
            "id": first, "token": vocab[first], "probability": probabilities[first].item(),
            "runner_up": {"id": second, "token": vocab[second], "probability": probabilities[second].item()},
        })
        if first == ids["<EOS>"]:
            return "".join(parts), trace, True
        parts.append(vocab[first])
        tokens = torch.cat((tokens, torch.tensor([[first]])), dim=1)
    return "".join(parts), trace, False


def evaluate(model, vocab, questions=None):
    # 默认读取评估文件，逐题保留文本、候选概率与停止状态。
    if questions is None:
        questions = load_eval_questions()
    results = []
    for question in questions:
        text, trace, stopped = predict(model, vocab, question)
        results.append({"question": question, "prediction": text, "trace": trace, "stopped_at_eos": stopped})
    return results
