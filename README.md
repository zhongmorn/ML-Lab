# ML-Lab
中文 | [English](README.en.md)

用 PyTorch 实现机器学习算法，记录实验的运行方法和结果。

<br>

## 目录

```text
ML-Lab
├── README.md
├── README.en.md
├── Deep-Learning
│   ├── 01.NeuralNetwork     # 神经网络
│   ├── 02.Regression        # 回归
│   ├── 03.Classification    # 分类
│   └── 04.Transformer       # 因果自注意力
└── Reinforcement-learning
    ├── 01.Q-learning          # 表格型 Q-learning
    ├── 02.DQN                 # 深度 Q 网络
    ├── 03.DDQN                # 双重深度 Q 网络
    └── 04.PPO                 # PPO-Clip + GAE
```

<br>

## 运行实验
从仓库根目录进入对应的实验目录。以下是运行回归实验的命令。

```bash
cd Deep-Learning/02.Regression
python -m pip install -r requirements.txt
python main.py
```

DQN 和 Double DQN 在各自的实验目录中运行 `python main.py`，环境在对应的 `main.py` 中选择。

运行 PPO 时，从仓库根目录执行以下命令。

```bash
cd Reinforcement-learning/04.PPO
python -m pip install -r requirements.txt
python main.py
```

在 PPO 的 `main.py` 中选择实验配置。

<br>

## 实验入口
### Deep-Learning

| 文件 | 名称 | 任务 |
| --- | --- | --- |
| [01.NeuralNetwork](Deep-Learning/01.NeuralNetwork/docs/README.md) | 正向传播与反向传播 | 合成数据 |
| [02.Regression](Deep-Learning/02.Regression/docs/README.md) | 回归模型 | 合成数据 |
| [03.Classification](Deep-Learning/03.Classification/docs/README.md) | 分类模型 | Fashion-MNIST |
| [04.Transformer](Deep-Learning/04.Transformer/docs/README.md) | Transformer | 合成语料 |

### Reinforcement-learning

| 文件 | 名称 | 任务 |
| --- | --- | --- |
| [01.Q-learning](Reinforcement-learning/01.Q-learning/docs/README.md) | 表格型 Q-learning | 在 8×8 打滑迷宫中找食物 |
| [02.DQN](Reinforcement-learning/02.DQN/docs/README.md) | 深度 Q 网络 | CartPole-v1 |
| [03.DDQN](Reinforcement-learning/03.DDQN/docs/README.md) | 双重深度 Q 网络 | CartPole-v1 |
| [04.PPO](Reinforcement-learning/04.PPO/docs/README.md) | 近端策略优化（PPO-Clip） | CartPole-v1、Pendulum-v1 |

实验原理和结果见各实验目录中的 `docs/README.md`。

<br>

## 未来计划

| 名称 |
| --- |
| 传统机器学习 |
| 时间序列 |
| 推荐与排序 |
| 图学习 |
| 异常检测 |
| 因果推断 |
| 多模态学习 |
| 自监督表征学习 |
| 图像生成 |
| 大模型后训练 |
| 模型压缩与推理优化 |
| 半监督学习 |
| 迁移学习 |
| 联邦学习 |
| 多智能体强化学习 |
