# ML-Lab
中文 | [English](README.en.md)

机器学习算法实现，记录各实验的运行方法和结果。

<br>

## 目录

```text
ML-Lab
├── README.md
├── README.en.md
├── Classical-ML
│   └── 01.XGBoost           # 极端梯度提升
├── Deep-Learning
│   ├── 01.NeuralNetwork     # 神经网络
│   ├── 02.Regression        # 回归
│   ├── 03.Classification    # 分类
│   └── 04.Transformer       # 因果自注意力
└── Reinforcement-learning
    ├── 01.Q-learning          # 表格型 Q-learning
    ├── 02.DQN                 # 深度 Q 网络
    ├── 03.DDQN                # 双重深度 Q 网络
    ├── 04.PPO                 # PPO-Clip + GAE
    ├── 05.GRPO                # 组相对策略优化
    ├── 06.RLHF                # 奖励模型 + PPO
    └── 07.DPO                 # 直接偏好优化
```

<br>

## 运行实验

从仓库根目录进入对应的实验目录。以下是运行回归实验的命令。

```bash
cd Deep-Learning/02.Regression
python -m pip install -r requirements.txt
python main.py
```

XGBoost 在 `Classical-ML/01.XGBoost` 中运行 `python -m pip install -r requirements.txt` 和 `python main.py`。

DQN 和 Double DQN 在各自的实验目录中运行 `python main.py`，训练参数在 `run/configs.json` 中设置。

运行 PPO 时，从仓库根目录执行以下命令。

```bash
cd Reinforcement-learning/04.PPO
python -m pip install -r requirements.txt
python main.py
```

在 PPO 的 `main.py` 中用 `ENV_ID` 选择环境，参数在 `run/configs.json` 的对应条目中设置。

GRPO、RLHF 和 DPO 在各自的实验目录中运行 `python main.py`。

<br>

## 实验入口

### Classical-ML

| 文件 | 名称 | 任务 |
| --- | --- | --- |
| [01.XGBoost](Classical-ML/01.XGBoost/docs/README.md) | XGBoost | 乳腺癌数据集 |

**01.XGBoost** (Chen & Guestrin, 2016 [Scalable Tree Boosting](https://arxiv.org/abs/1603.02754))

<br>

### Deep-Learning

| 文件 | 名称 | 任务 |
| --- | --- | --- |
| [01.NeuralNetwork](Deep-Learning/01.NeuralNetwork/docs/README.md) | 正向传播与反向传播 | 合成数据 |
| [02.Regression](Deep-Learning/02.Regression/docs/README.md) | 回归模型 | 合成数据 |
| [03.Classification](Deep-Learning/03.Classification/docs/README.md) | 分类模型 | Fashion-MNIST |
| [04.Transformer](Deep-Learning/04.Transformer/docs/README.md) | Transformer | 合成语料 |

**01.NeuralNetwork** (Rumelhart et al., 1986 [Backpropagating Errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf))

**02.Regression** (Yarotsky, 2017 [Deep ReLU Approximation](https://arxiv.org/abs/1610.01145))

**03.Classification** (Kingma & Ba, 2015 [Adam](https://arxiv.org/abs/1412.6980)) (Xiao et al., 2017 [Fashion-MNIST](https://arxiv.org/abs/1708.07747))

**04.Transformer** (Vaswani et al., 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762))

<br>

### Reinforcement-learning

| 文件 | 名称 | 任务 |
| --- | --- | --- |
| [01.Q-learning](Reinforcement-learning/01.Q-learning/docs/README.md) | 表格型 Q-learning | 在 8×8 打滑迷宫中找食物 |
| [02.DQN](Reinforcement-learning/02.DQN/docs/README.md) | 深度 Q 网络 | CartPole-v1 |
| [03.DDQN](Reinforcement-learning/03.DDQN/docs/README.md) | 双重深度 Q 网络 | CartPole-v1 |
| [04.PPO](Reinforcement-learning/04.PPO/docs/README.md) | 近端策略优化（PPO-Clip） | CartPole-v1、Pendulum-v1 |
| [05.GRPO](Reinforcement-learning/05.GRPO/docs/README.md) | 组相对策略优化 | 四动作任务，使用组内奖励 |
| [06.RLHF](Reinforcement-learning/06.RLHF/docs/README.md) | 基于人类反馈的强化学习 | 四动作任务，合成偏好与奖励模型 |
| [07.DPO](Reinforcement-learning/07.DPO/docs/README.md) | 直接偏好优化 | 四动作任务，固定偏好对 |

**01.Q-learning** (Watkins & Dayan, 1992 [Q-learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf))

**02.DQN** (Mnih et al., 2015 [Human-level Control](https://deepmind-media.storage.googleapis.com/dqn/DQNNaturePaper.pdf))

**03.DDQN** (van Hasselt et al., 2016 [Deep Double Q-learning](https://arxiv.org/abs/1509.06461))

**04.PPO** (Schulman et al., 2017 [PPO](https://arxiv.org/abs/1707.06347)) (Schulman et al., 2016 [GAE](https://arxiv.org/abs/1506.02438))

**05.GRPO** (Shao et al., 2024 [DeepSeekMath](https://arxiv.org/abs/2402.03300))

**06.RLHF** (Ouyang et al., 2022 [InstructGPT](https://arxiv.org/abs/2203.02155))

**07.DPO** (Rafailov et al., 2023 [DPO](https://arxiv.org/abs/2305.18290))

实验原理和结果见各实验目录中的 `docs/README.md`。

<br>

## 未来计划

| 名称 |
| --- |
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
