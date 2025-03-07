import random
import torch
import numpy as np

from collections import namedtuple, deque
from typing import Generator

from Snakes.UCT.Network import SnakeNet
from Snakes.UCT.UCT import Root
from Constants import DEVICE, MEMORY_SIZE, BATCH_SIZE, EPOCHS


def get_symetries(
    conv: torch.Tensor, aid: torch.Tensor, possible_actions: torch.Tensor, policy: torch.Tensor
) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    yield (conv, aid, possible_actions, policy)

    yield (
        torch.flip(conv, dims=(1,)),
        torch.cat((aid[:2], torch.flip(aid[2:], dims=(0,)))),
        torch.flip(possible_actions, dims=(0,)),
        torch.flip(policy, dims=(0,)),
    )


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, convs, aids, possible_actions, policies, winner, loss_factor):
        """Save a transition"""
        self.memory.append(Transition(convs, aids, possible_actions, policies, winner, loss_factor))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class GameMemory(object):
    def __init__(self):
        self.moves: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        """convs, aids, possible_actions, policies, turn"""

    def save_move(
        self,
        root: Root,
        main_action_values: tuple[float, float, float],
        other_action_values: tuple[float, float, float],
        turn: int,
    ):
        self.moves.append(
            (
                *root.board.to_tensors(DEVICE),
                torch.stack(
                    (
                        torch.softmax(torch.tensor(main_action_values, dtype=torch.float, device=DEVICE), dim=0),
                        torch.softmax(-torch.tensor(other_action_values, dtype=torch.float, device=DEVICE), dim=0),
                    )
                ),
                turn,
            )
        )

    def save_into_memory(self, memory: ReplayMemory, winner: float, max_turn: int):
        for convs, aids, possible_actions, policies, turn in self.moves:
            for i in range(len(convs)):
                for symetric_conv, symetric_aid, symetric_possible_actions, symetric_policy in get_symetries(
                    convs[i], aids[i], possible_actions[i], policies[i]
                ):
                    memory.push(
                        symetric_conv,
                        symetric_aid,
                        symetric_possible_actions,
                        symetric_policy,
                        torch.tensor(winner, dtype=torch.float, device=DEVICE),
                        torch.tensor(np.exp(-0.05 * (max_turn - turn)), dtype=torch.float, device=DEVICE),
                    )

    def __len__(self):
        return len(self.moves)


Transition = namedtuple("Transition", ("convs", "aids", "possible_actions", "policies", "winner", "loss_factor"))
memory = ReplayMemory(MEMORY_SIZE)


def train(model: SnakeNet, memory: ReplayMemory, optimizer) -> float:
    model.train()

    cumul_loss = 0.0
    criterion = torch.nn.MSELoss(reduction="none")

    for epoch in range(EPOCHS):
        batch = Transition(*zip(*memory.sample(BATCH_SIZE)))

        batch_convs = torch.stack(batch.convs)
        batch_aids = torch.stack(batch.aids)
        batch_possible_actions = torch.stack(batch.possible_actions)
        batch_policies = torch.stack(batch.policies)
        batch_winner = torch.tensor(batch.winner)
        batch_loss_factor = torch.tensor(batch.loss_factor)

        optimizer.zero_grad()
        pred_winner, pred_policies = model(batch_convs, batch_aids, batch_possible_actions)

        loss_winner: torch.Tensor = criterion(pred_winner, batch_winner) * batch_loss_factor
        loss_policies: torch.Tensor = criterion(pred_policies, batch_policies) * batch_loss_factor.view(-1, 1)

        loss = loss_winner.mean() + loss_policies.mean()

        loss.backward()
        optimizer.step()

        cumul_loss += loss.item()

    return cumul_loss / EPOCHS
