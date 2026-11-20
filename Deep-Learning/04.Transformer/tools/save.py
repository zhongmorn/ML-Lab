import json
from pathlib import Path

import torch


def save_experiment(checkpoint, model, vocab, report):
    # 先完整写入临时文件，再替换检查点；权重同时保存词表和模型尺寸。
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "model_config": report["model_config"]}, temporary)
    temporary.replace(checkpoint)
    # 在权重旁保存含中文的 JSON 实验报告。
    report_path = checkpoint.with_name("results.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
