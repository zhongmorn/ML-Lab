from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class DPO:
    def __init__(self, config):
        self.beta = config["beta"]
        self.policy = nn.Linear(2, 4, device="cpu")
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)

        # 冻结初始策略，作为每个偏好对的比较基准。
        self.reference = deepcopy(self.policy).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config["learning_rate"])

    def preference_loss(self, contexts, chosen, rejected):
        # 两列分别对应优选和劣选动作。
        actions = torch.stack((chosen, rejected), dim=1)
        policy_log_probs = self.policy(contexts).log_softmax(dim=-1).gather(1, actions)
        with torch.no_grad():
            reference_log_probs = self.reference(contexts).log_softmax(dim=-1).gather(1, actions)

        # 从策略的偏好差值中扣除参考策略的偏好差值。
        policy_margin = policy_log_probs[:, 0] - policy_log_probs[:, 1]
        reference_margin = reference_log_probs[:, 0] - reference_log_probs[:, 1]
        return -F.logsigmoid(self.beta * (policy_margin - reference_margin)).mean()

    def update(self, contexts, chosen, rejected):
        loss = self.preference_loss(contexts, chosen, rejected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
