import json

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

from .data import load_data
from tools.plot import plot_confusion_matrix, plot_feature_importance, plot_training


def run_experiment(config, output_dir):
    # 只在训练集拟合模型；验证集记录学习过程，测试集用于最终评估。
    output_dir.mkdir(parents=True, exist_ok=True)
    features, labels, feature_names, splits = load_data(config["random_state"])
    train, validation = splits["train"], splits["validation"]
    model = XGBClassifier(**config)
    model.fit(
        features[train], labels[train],
        eval_set=[(features[train], labels[train]), (features[validation], labels[validation])],
        verbose=False,
    )

    # 记录每一轮的训练与验证损失，以及对应树数量下的分类准确率。
    history = model.evals_result()
    training_rows = []
    for round_number in range(1, config["n_estimators"] + 1):
        train_probability = model.predict_proba(features[train], iteration_range=(0, round_number))[:, 1]
        validation_probability = model.predict_proba(features[validation], iteration_range=(0, round_number))[:, 1]
        training_rows.append({
            "round": round_number,
            "train_logloss": history["validation_0"]["logloss"][round_number - 1],
            "validation_logloss": history["validation_1"]["logloss"][round_number - 1],
            "train_accuracy": float(accuracy_score(labels[train], train_probability >= 0.5)),
            "validation_accuracy": float(accuracy_score(labels[validation], validation_probability >= 0.5)),
        })

    # 正类概率表示恶性类别的概率；所有数据集都使用固定的 0.5 阈值。
    metrics = {}
    for name, indices in splits.items():
        probability = model.predict_proba(features[indices])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        target = labels[indices]
        metrics[name] = {
            "samples": len(indices),
            "correct": int(np.sum(prediction == target)),
            "accuracy": float(accuracy_score(target, prediction)),
            "precision": float(precision_score(target, prediction, zero_division=0)),
            "recall": float(recall_score(target, prediction, zero_division=0)),
            "f1": float(f1_score(target, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(target, probability)),
            "log_loss": float(log_loss(target, probability)),
        }

    # 根据测试集预测计算混淆矩阵。
    test = splits["test"]
    test_probability = model.predict_proba(features[test])[:, 1]
    test_prediction = (test_probability >= 0.5).astype(int)
    matrix = confusion_matrix(labels[test], test_prediction, labels=[0, 1])

    # gain 重要性为各特征平均分裂增益归一化后的数值。
    importance_rows = [
        {"feature": str(name), "importance": float(value)}
        for name, value in zip(feature_names, model.feature_importances_)
    ]
    importance_rows.sort(key=lambda row: row["importance"], reverse=True)

    # 只保存指标汇总和三张结果图。
    summary = {
        "algorithm": "XGBoost",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "parameters": config,
        "metrics": metrics,
        "test_confusion_matrix": matrix.tolist(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_training(training_rows, output_dir / "training.png")
    plot_confusion_matrix(matrix, output_dir / "confusion_matrix.png")
    plot_feature_importance(importance_rows, output_dir / "feature_importance.png")
    print(json.dumps({"parameters": config, "metrics": metrics,
                      "test_confusion_matrix": matrix.tolist()}, indent=2))
    return model, summary
