"""统一入口：python main.py train / evaluate / plot。"""

import argparse
import json
from pathlib import Path
import sys

from run import PPO, PPOConfig
from tools.plot import plot_training
from output.report import update_readme_results
from tools.save import write_json, write_evaluation_episodes

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO training, evaluation and plotting")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train PPO and save Train/Eval/Loss figures")
    train.add_argument("--config", type=Path, default=ROOT / "run/config/cartpole.json")
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--run-number", type=int, default=None,
                       help="Optional repetition number; a single experiment uses seedN")
    train.add_argument("--no-plots", action="store_true", help="Only save logs; draw later")
    train.add_argument("--env-id", default=argparse.SUPPRESS)
    train.add_argument("--device", default=argparse.SUPPRESS)
    for name in ("train-rounds", "batch-size", "update-epochs", "final-eval-episodes",
                 "final-eval-seed", "eval-batch-size", "seed", "torch-threads"):
        train.add_argument("--" + name, type=int, default=argparse.SUPPRESS)
    for name in ("learning-rate", "reward-scale", "gamma", "gae-lambda", "clip-coef", "value-coef",
                 "entropy-coef", "max-grad-norm", "target-kl"):
        train.add_argument("--" + name, type=float, default=argparse.SUPPRESS)
    train.add_argument("--hidden-sizes", type=int, nargs="+", default=argparse.SUPPRESS)
    train.add_argument("--monitor-seeds", type=int, nargs=3, default=argparse.SUPPRESS)
    train.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    train.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    evaluate = commands.add_parser("evaluate", help="Load and evaluate a checkpoint")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--episodes", type=int, default=100)
    evaluate.add_argument("--seed", type=int, default=30_000)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--eval-batch-size", type=int, default=32)
    evaluate.add_argument("--stochastic", action="store_true")
    evaluate.add_argument("--render", action="store_true")
    evaluate.add_argument("--output", type=Path)

    plot = commands.add_parser("plot", help="Draw Train/Eval/Loss")
    plot.add_argument("--run-dir", type=Path, required=True)
    plot.add_argument("--window", type=int, default=20)
    plot.add_argument("--output-dir", type=Path)
    return parser


def train_command(values: dict) -> None:
    config_path = values.pop("config")
    output_dir = values.pop("output_dir")
    no_plots = values.pop("no_plots")
    run_number = values.pop("run_number", None)
    if run_number is not None and (not isinstance(run_number, int) or isinstance(run_number, bool) or run_number <= 0):
        raise ValueError("run_number must be a positive integer")
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config = PPOConfig(**{**config_data, **values})
    if output_dir is None:
        environment_dir = ROOT / "output" / config.env_id.replace("/", "_")
        run_name = f"seed{config.seed}" if run_number is None else f"seed{config.seed}_run{run_number}"
        output_dir = environment_dir / run_name
    summary = PPO(config).train(output_dir)
    if not no_plots:
        plot_training(output_dir)
    update_readme_results()
    print(json.dumps({"output_dir": str(output_dir.resolve()),
                      "best_mean_return": summary["best_mean_return"],
                      "elapsed_seconds": round(summary["elapsed_seconds"], 2)}, ensure_ascii=False, indent=2))


def evaluate_command(values: dict) -> None:
    checkpoint = values["checkpoint"]
    agent = PPO.load(checkpoint, device=values["device"])
    if values["eval_batch_size"] <= 0:
        raise ValueError("eval_batch_size must be positive")
    agent.config.eval_batch_size = values["eval_batch_size"]
    result = agent.evaluate(
        episodes=values["episodes"], seed=values["seed"],
        deterministic=not values["stochastic"], render=values["render"])
    result_dir = checkpoint.parent.parent / "logs" if checkpoint.parent.name == "model" else checkpoint.parent
    policy = "stochastic" if values["stochastic"] else "deterministic"
    output = values["output"] or result_dir / f"{checkpoint.stem}_evaluation_seed{values['seed']}_{policy}.json"
    if output.resolve() == checkpoint.resolve():
        raise ValueError("Evaluation output must not overwrite the checkpoint")
    write_json(output, result)
    write_evaluation_episodes(result_dir / "evaluation_episodes.csv", result, checkpoint=checkpoint.name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Evaluation saved to {output.resolve()}")


def main() -> None:
    parser = build_parser()
    # 直接 python main.py 采用默认 CartPole 训练配置。
    args = vars(parser.parse_args(sys.argv[1:] or ["train"]))
    command = args.pop("command")
    try:
        if command == "train":
            train_command(args)
        elif command == "evaluate":
            evaluate_command(args)
        else:
            paths = plot_training(args["run_dir"], args["window"], args["output_dir"])
            update_readme_results()
            for label, path in paths.items():
                print(f"Saved {label}: {path.resolve()}")
    except (ValueError, TypeError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
