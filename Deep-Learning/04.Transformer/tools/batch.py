import torch

from functools import lru_cache


# 按首字符缓存候选词元，排除控制标记并按长度降序，供最长匹配编码。
@lru_cache(maxsize=8)
def _prefix_index(vocab):
    by_first = {}
    for index, piece in enumerate(vocab):
        if piece and not (piece.startswith("<") and piece.endswith(">")):
            by_first.setdefault(piece[0], []).append((piece, index))
    for candidates in by_first.values():
        candidates.sort(key=lambda item: len(item[0]), reverse=True)
    return by_first


# 从左到右最长匹配；陌生字符使用 UNK，旧词表没有 UNK 时报告位置。
def encode(text, vocab):
    index = _prefix_index(tuple(vocab))
    ids = []
    offset = 0
    while offset < len(text):
        candidates = index.get(text[offset], ())
        match = next((item for item in candidates if text.startswith(item[0], offset)), None)
        if match is None:
            if "<UNK>" not in vocab:
                raise ValueError(f"词表无法识别位置 {offset} 的字符 {text[offset]!r}")
            ids.append(vocab.index("<UNK>"))
            offset += 1
        else:
            piece, token_id = match
            ids.append(token_id)
            offset += len(piece)
    return ids


def language_model_rows(texts, vocab, context_size):
    # 各段文本独立加入 BOS/EOS；相邻窗口共享一个边界词元，以保留下一词目标。
    ids = {token: i for i, token in enumerate(vocab)}
    rows = []
    for text in texts:
        sequence = [ids["<BOS>"], *encode(text, vocab), ids["<EOS>"]]
        for start in range(0, len(sequence) - 1, context_size):
            rows.append(sequence[start:start + context_size + 1])
    # 窗口补齐后将输入和目标错开一位；PAD 目标置为 -100，不参与交叉熵。
    padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(row, dtype=torch.long) for row in rows],
        batch_first=True,
        padding_value=ids["<PAD>"],
    )
    inputs, targets = padded[:, :-1], padded[:, 1:]
    targets = targets.masked_fill(targets == ids["<PAD>"], -100)
    return inputs, targets
