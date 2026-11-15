# ML-Lab

用 PyTorch 实现机器学习和强化学习算法，记录如何运行实验以及跑出了什么结果。

## 目录

```text
ML-Lab
├── README.md
├── CHANGELOG.md
├── Machine-Learning
│   ├── Regression          # 回归
│   ├── Classification      # 分类
│   └── NeuralNetwork       # 神经网络
└── Reinforcement-learning
    ├── DQN                 # 深度 Q 网络
    ├── DoubleDQN           # 双重深度 Q 网络
    ├── OPE                 # 离线策略评估
    └── PPO                 # PPO-Clip + GAE
```

## 运行实验

从仓库根目录进入你要运行的实验。例如，运行回归实验：

```bash
cd Machine-Learning/Regression
python -m pip install -r requirements.txt
python main.py
```

运行 OPE 时，先进入对应的环境目录。运行 PPO 时，从仓库根目录执行：

```bash
cd Reinforcement-learning/PPO
python -m pip install -r requirements.txt
python main.py train --config run/config/cartpole.json --device cpu --run-number 1
```

把配置换成 `run/config/pendulum.json` 可以运行 Pendulum 实验。有可用的 CUDA 环境时，可将设备参数改为 `cuda`。

## 实验入口

### Machine-Learning

| 算法 | 实验 |
| --- | --- |
| [回归](Machine-Learning/Regression/README.md) | 合成数据 |
| [分类](Machine-Learning/Classification/README.md) | Fashion-MNIST |
| [神经网络](Machine-Learning/NeuralNetwork/README.md) | 正向传播与反向传播 |

### Reinforcement-learning

| 实验 | 任务 |
| --- | --- |
| [DQN](Reinforcement-learning/DQN/README.md) | CartPole-v1、MountainCar-v0 |
| [DoubleDQN](Reinforcement-learning/DoubleDQN/README.md) | CartPole-v1、MountainCar-v0 |
| [OPE](Reinforcement-learning/OPE/README.md) | Pendulum-v1、Swimmer-v5 |
| [PPO](Reinforcement-learning/PPO/README.md) | CartPole-v1、Pendulum-v1 |

实验原理与代码放在同一目录。
