import json

import numpy as np

from .configs import CONFIG
from environment import gridworld
from tools.plot import plot_results

from .agent import QLearning
from .baseline import shortest_path_policy


TRAINING = CONFIG["training"]
EVALUATION = CONFIG["evaluation"]
EPISODES = TRAINING["episodes"]
MAX_STEPS = CONFIG["maze"]["max_steps"]
EVALUATION_EPISODES = EVALUATION["episodes"]
EVALUATION_SEED = EVALUATION["environment_seed"]
ROLLING_WINDOW = CONFIG["reporting"]["rolling_window"]


def run_episode(select_action, environment_rng, agent=None, *, step_reward=gridworld.STEP_REWARD):
    state = gridworld.START
    path = [state]
    total_reward = 0.0
    outcome = "timeout"

    for _ in range(MAX_STEPS):
        action = select_action(state)
        next_state, reward, terminated = gridworld.step(state, action, environment_rng, step_reward=step_reward)
        # 用环境实际返回的奖励，更新打滑前所选动作的 Q 值。
        if agent is not None:
            agent.update(state, action, reward, next_state, terminated)
        total_reward += reward
        path.append(next_state)
        state = next_state
        if terminated:
            outcome = "food" if state == gridworld.GOAL else "hazard"
            break

    return {
        "steps": len(path) - 1,
        "return": round(total_reward, 6),
        "outcome": outcome,
        "path": path,
    }


def evaluate(select_action, *, step_reward=gridworld.STEP_REWARD):
    results = []
    # 三种策略共用一组环境种子，用相同的打滑随机数序列评估。
    for seed in range(EVALUATION_SEED, EVALUATION_SEED + EVALUATION_EPISODES):
        result = run_episode(select_action, np.random.default_rng(seed), step_reward=step_reward)
        results.append({"environment_seed": seed, **result})
    return results


def summarize(results):
    # 平均成功步数只统计到达食物的回合。
    success_steps = [result["steps"] for result in results if result["outcome"] == "food"]
    return {
        "successes": len(success_steps),
        "hazards": sum(result["outcome"] == "hazard" for result in results),
        "timeouts": sum(result["outcome"] == "timeout" for result in results),
        "success_rate": len(success_steps) / len(results),
        "mean_return": round(sum(result["return"] for result in results) / len(results), 6),
        "mean_success_steps": round(sum(success_steps) / len(success_steps), 2) if success_steps else None,
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_experiment(output_dir, *, step_reward=gridworld.STEP_REWARD):
    agent = QLearning(TRAINING["learning_rate"], TRAINING["discount"])
    training_rng = np.random.default_rng(TRAINING["environment_seed"])
    rows = []

    for episode in range(1, EPISODES + 1):
        result = run_episode(agent.select_action, training_rng, agent, step_reward=step_reward)
        rows.append({
            "episode": episode,
            "steps": result["steps"],
            "return": result["return"],
            "success": int(result["outcome"] == "food"),
            "outcome": result["outcome"],
        })

    baseline_rng = np.random.default_rng(EVALUATION["random_action_seed"])
    policies = {
        "random": lambda state: int(baseline_rng.integers(len(gridworld.MOVES))),
        "shortest_path": shortest_path_policy(),
        "q_learning": agent.select_action,
    }
    runs = {name: evaluate(policy, step_reward=step_reward) for name, policy in policies.items()}
    metrics = {name: summarize(results) for name, results in runs.items()}
    evaluation_rows = [
        {
            "policy": name,
            "environment_seed": result["environment_seed"],
            "steps": result["steps"],
            "return": result["return"],
            "outcome": result["outcome"],
        }
        for name, results in runs.items()
        for result in results
    ]
    # 优先画第一个成功回合，全部失败时画第一个回合。
    greedy_runs = runs["q_learning"]
    example = next((result for result in greedy_runs if result["outcome"] == "food"), greedy_runs[0])
    recent_rows = rows[-ROLLING_WINDOW:]

    train_log = {
        f"last_{ROLLING_WINDOW}_success_rate": sum(row["success"] for row in recent_rows) / len(recent_rows),
        "records": rows,
    }
    eval_log = {
        "policies": metrics,
        "greedy_example": example,
        "records": evaluation_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(rows, example["path"], output_dir)
    write_json(output_dir / "results.json", {"training": train_log, "evaluation": eval_log})

    print(
        f"{gridworld.SIZE}×{gridworld.SIZE} 打滑迷宫，训练 {EPISODES} 回合，"
        f"每种策略评估 {EVALUATION_EPISODES} 回合"
    )
    for name, label in (("random", "随机策略"), ("shortest_path", "最短路策略"), ("q_learning", "Q-learning")):
        result = metrics[name]
        print(
            f"{label}  成功 {result['success_rate']:.1%}  "
            f"踩陷阱 {result['hazards']} 次  超时 {result['timeouts']} 次  平均回报 {result['mean_return']:.3f}"
        )
    print(f"实验结果保存在 {output_dir}")
