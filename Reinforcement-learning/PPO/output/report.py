"""只更新根 README 的实验结果区，保留数学理论与目录说明。"""

import json
from pathlib import Path
from urllib.parse import quote

START = "<!-- EXPERIMENT_RESULTS_START -->"
END = "<!-- EXPERIMENT_RESULTS_END -->"


def update_readme_results(project_root: str | Path | None = None) -> Path:
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    readme = root / "README.md"
    records = []
    for summary_path in sorted((root / "output").rglob("summary.json")):
        config_path = summary_path.with_name("config.json")
        if not config_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (summary.get("protocol") == "episode_rounds_fixed_three_seeds"
                and "final_evaluation" in summary and "env_id" in config):
            records.append((summary_path.parent, config, summary))

    lines = []
    for run_dir, config, _ in records:
        figures = [(label, run_dir / "figures" / filename) for label, filename in (
            ("Train", "training_reward.png"), ("Eval", "evaluation_reward.png"), ("Loss", "loss.png"))]
        if not any(path.is_file() for _, path in figures):
            continue
        cells = []
        for label, path in figures:
            if not path.is_file():
                cells.append("—")
                continue
            target = quote(path.relative_to(root).as_posix(), safe='/._-')
            cells.append(f"![{label}]({target})")
        lines.extend([
            f"### {config['env_id']} / {run_dir.name}", "",
            "| Train | Eval | Loss |", "|:---:|:---:|:---:|",
            "| " + " | ".join(cells) + " |", "",
        ])
    content = readme.read_text(encoding="utf-8") if readme.exists() else "# PPO\n"
    block = START + "\n\n" + "\n".join(lines) + "\n" + END
    if START in content and END in content and content.index(START) < content.index(END):
        before, tail = content.split(START, 1)
        _, after = tail.split(END, 1)
        content = before + block + after
    else:
        content = content.rstrip() + "\n\n## 实验结果汇总\n\n" + block + "\n"
    readme.write_text(content, encoding="utf-8")
    return readme
