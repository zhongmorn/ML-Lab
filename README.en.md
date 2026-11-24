# ML-Lab
[中文](README.md) | English

PyTorch implementations of machine learning algorithms, with instructions and results for each experiment.

<br>

## Repository Structure

```text
ML-Lab
├── README.md
├── README.en.md
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

For DQN and Double DQN, run `python main.py` from the corresponding experiment directory. Select the environment in that directory's `main.py`.

For PPO, run these commands from the repository root.

```bash
cd Reinforcement-learning/04.PPO
python -m pip install -r requirements.txt
python main.py
```

Select the PPO configuration in `main.py`.

For GRPO, RLHF, and DPO, run `python main.py` from the corresponding experiment directory.

<br>

## Experiments
### Deep-Learning

| Folder | Name | Task |
| --- | --- | --- |
| [01.NeuralNetwork](Deep-Learning/01.NeuralNetwork/docs/README.en.md) | Forward Propagation and Backpropagation | Synthetic data |
| [02.Regression](Deep-Learning/02.Regression/docs/README.en.md) | Regression Models | Synthetic data |
| [03.Classification](Deep-Learning/03.Classification/docs/README.en.md) | Classification Model | Fashion-MNIST |
| [04.Transformer](Deep-Learning/04.Transformer/docs/README.en.md) | Transformer | Synthetic corpus |

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

See `docs/README.en.md` in each experiment directory for the algorithm and results.

<br>

## Future Plans

| Name |
| --- |
| Classical Machine Learning |
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
