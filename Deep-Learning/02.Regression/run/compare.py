import json
from pathlib import Path

import torch

from .data import get_data_loader, synthetic_data
from .model import DeepRegressionModel, LinearRegressionModel
from .train import train_model
from tools.plot import plot_comparison

# 对照线性目标、非线性目标和高噪声非线性目标。
DATASETS = (
    ("Linear target", False, 0.01),
    ("Nonlinear target", True, 0.01),
    ("High-noise nonlinear", True, 1.0),
)
MODELS = (
    ("Linear", LinearRegressionModel),
    ("Deep", DeepRegressionModel),
)


def compare(output_dir: Path):
    weights = torch.tensor([2.0, -3.4])
    bias = 4.2
    results = {}

    for dataset, nonlinear, noise_std in DATASETS:
        # 各条件复用相同的特征抽样；训练集与验证集使用独立随机种子。
        train_data = synthetic_data(
            weights, bias, 2048, nonlinear=nonlinear, noise_std=noise_std,
            generator=torch.Generator().manual_seed(42),
        )
        validation_features, validation_labels = synthetic_data(
            weights, bias, 500, nonlinear=nonlinear, noise_std=noise_std,
            generator=torch.Generator().manual_seed(43),
        )
        results[dataset] = {}

        for name, model_type in MODELS:
            print(f"\n{dataset} / {name}")
            # 固定初始化和批次顺序，使同一模型在各条件下可比。
            torch.manual_seed(44)
            model = model_type(len(weights))
            loader = get_data_loader(
                *train_data, batch_size=32, generator=torch.Generator().manual_seed(45),
            )
            losses = train_model(model, loader, epochs=100, lr=0.01)

            # 独立验证集上计算 MAE 与 RMSE，训练损失留作收敛曲线。
            model.eval()
            with torch.no_grad():
                errors = model(validation_features) - validation_labels
                results[dataset][name] = {
                    "losses": losses,
                    "mae": errors.abs().mean().item(),
                    "rmse": errors.square().mean().sqrt().item(),
                }

    # 保存各条件的训练曲线和验证误差对照图。
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, output_dir)

    # 保存各组末轮训练 MSE、验证 MAE 和 RMSE。
    records = []
    print("\nDataset / model: training MSE, validation MAE, validation RMSE")
    for dataset, models in results.items():
        for name, metrics in models.items():
            train_mse = metrics["losses"][-1]
            mae, rmse = metrics["mae"], metrics["rmse"]
            records.append({
                "dataset": dataset,
                "model": name,
                "train_mse": train_mse,
                "validation_mae": mae,
                "validation_rmse": rmse,
            })
            print(f"{dataset} / {name}: {train_mse:.6f}, {mae:.6f}, {rmse:.6f}")

    (output_dir / "results.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8",
    )

    print(f"\nSaved results to {output_dir}")
