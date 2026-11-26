# ML-Lab
[中文](README.md) | English

Machine learning algorithm implementations, with instructions and results for each experiment.

<br>

## Repository Structure

```text
ML-Lab
├── README.md
├── README.en.md
├── Classical-ML
│   └── 01.XGBoost           # Extreme Gradient Boosting
├── Deep-Learning
│   ├── 01.NeuralNetwork     # Neural network
│   ├── 02.Regression        # Regression
│   ├── 03.Classification    # Classification
│   └── 04.Transformer       # Causal self-attention
└── Reinforcement-learning
    ├── 01.Q-learning          # Tabular Q-learning
    ├── 02.DQN                 # Deep Q-network
    ├── 03.DDQN                # Double DQN
    ├── 04.PPO                 # PPO-Clip + GAE
    ├── 05.GRPO                # Group Relative Policy Optimization
    ├── 06.RLHF                # Reward model + PPO
    └── 07.DPO                 # Direct Preference Optimization
```

<br>

## Running the Experiments

Start from the repository root and open the directory for the experiment you want to run. To run the regression experiment, use the commands below.

```bash
cd Deep-Learning/02.Regression
python -m pip install -r requirements.txt
python main.py
```

For XGBoost, run `python -m pip install -r requirements.txt` and `python main.py` from `Classical-ML/01.XGBoost`.

For DQN and Double DQN, run `python main.py` from the corresponding experiment directory. Set training parameters in `run/configs.json`.

For PPO, run these commands from the repository root.

```bash
cd Reinforcement-learning/04.PPO
python -m pip install -r requirements.txt
python main.py
```

Set `ENV_ID` in PPO's `main.py` to select the environment. Edit that environment's settings in `run/configs.json`.

For GRPO, RLHF, and DPO, run `python main.py` from the corresponding experiment directory.

<br>

## Experiments

### Classical-ML

| Folder | Name | Task |
| --- | --- | --- |
| [01.XGBoost](Classical-ML/01.XGBoost/docs/README.en.md) | XGBoost | Breast Cancer Wisconsin (Diagnostic) |

**01.XGBoost** (Chen & Guestrin, 2016 [Scalable Tree Boosting](https://arxiv.org/abs/1603.02754))

<br>

### Deep-Learning

| Folder | Name | Task |
| --- | --- | --- |
| [01.NeuralNetwork](Deep-Learning/01.NeuralNetwork/docs/README.en.md) | Forward Propagation and Backpropagation | Synthetic data |
| [02.Regression](Deep-Learning/02.Regression/docs/README.en.md) | Regression Models | Synthetic data |
| [03.Classification](Deep-Learning/03.Classification/docs/README.en.md) | Classification Model | Fashion-MNIST |
| [04.Transformer](Deep-Learning/04.Transformer/docs/README.en.md) | Transformer | Synthetic corpus |

**01.NeuralNetwork** (Rumelhart et al., 1986 [Backpropagating Errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf))

**02.Regression** (Yarotsky, 2017 [Deep ReLU Approximation](https://arxiv.org/abs/1610.01145))

**03.Classification** (Kingma & Ba, 2015 [Adam](https://arxiv.org/abs/1412.6980)) (Xiao et al., 2017 [Fashion-MNIST](https://arxiv.org/abs/1708.07747))

**04.Transformer** (Vaswani et al., 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762))

<br>

### Reinforcement-learning

| Folder | Name | Task |
| --- | --- | --- |
| [01.Q-learning](Reinforcement-learning/01.Q-learning/docs/README.en.md) | Tabular Q-learning | Find food in an 8×8 maze with slips |
| [02.DQN](Reinforcement-learning/02.DQN/docs/README.en.md) | Deep Q-Network | CartPole-v1 |
| [03.DDQN](Reinforcement-learning/03.DDQN/docs/README.en.md) | Double Deep Q-Network | CartPole-v1 |
| [04.PPO](Reinforcement-learning/04.PPO/docs/README.en.md) | Proximal Policy Optimization (PPO-Clip) | CartPole-v1, Pendulum-v1 |
| [05.GRPO](Reinforcement-learning/05.GRPO/docs/README.en.md) | Group Relative Policy Optimization | Four-action task with group rewards |
| [06.RLHF](Reinforcement-learning/06.RLHF/docs/README.en.md) | Reinforcement Learning from Human Feedback | Four-action task with synthetic preferences and a reward model |
| [07.DPO](Reinforcement-learning/07.DPO/docs/README.en.md) | Direct Preference Optimization | Four-action task with fixed preference pairs |

**01.Q-learning** (Watkins & Dayan, 1992 [Q-learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf))

**02.DQN** (Mnih et al., 2015 [Human-level Control](https://deepmind-media.storage.googleapis.com/dqn/DQNNaturePaper.pdf))

**03.DDQN** (van Hasselt et al., 2016 [Deep Double Q-learning](https://arxiv.org/abs/1509.06461))

**04.PPO** (Schulman et al., 2017 [PPO](https://arxiv.org/abs/1707.06347)) (Schulman et al., 2016 [GAE](https://arxiv.org/abs/1506.02438))

**05.GRPO** (Shao et al., 2024 [DeepSeekMath](https://arxiv.org/abs/2402.03300))

**06.RLHF** (Ouyang et al., 2022 [InstructGPT](https://arxiv.org/abs/2203.02155))

**07.DPO** (Rafailov et al., 2023 [DPO](https://arxiv.org/abs/2305.18290))

See `docs/README.en.md` in each experiment directory for the algorithm and results.

<br>

## Future Plans

| Name |
| --- |
| Time Series |
| Recommendation and Ranking |
| Graph Learning |
| Anomaly Detection |
| Causal Inference |
| Multimodal Learning |
| Self-Supervised Representation Learning |
| Image Generation |
| LLM Post-Training |
| Model Compression and Inference Optimization |
| Semi-Supervised Learning |
| Transfer Learning |
| Federated Learning |
| Multi-Agent Reinforcement Learning |
