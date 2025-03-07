import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

from Constants import AID_SHAPE, DEVICE, VIEW_SIZE


class Evaluation(ABC):
    @abstractmethod
    def __call__(
        self, conv: torch.Tensor, aid: torch.Tensor, possible_actions: torch.Tensor, temperature: float = 1
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class Network(nn.Module, Evaluation): ...


class SnakeNet(nn.Module):
    """
    Evaluate how a single snake is doing

    Parameters:
        - (Batch, *CONV_SHAPE) for the view of the snake
        - (Batch, *AID_SHAPE) for the helpful parameters of the snake

    Returns 2 tensors of shape:
        - (Batch) for the score of the snake
        - (Batch, 3) values for the snake's policy
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, 3)  # 8x5x5
        self.conv2 = nn.Conv2d(8, 16, 3)  # 16x3x3
        self.conv3 = nn.Conv2d(16, 32, 3)  # 32x1x1

        self.conv_dense = nn.Linear(32 * (2 * VIEW_SIZE - 5) * (2 * VIEW_SIZE - 5), 32)

        self.dense1 = nn.Linear(32 + AID_SHAPE[1], 16)
        self.dense2 = nn.Linear(16, 4)

    def __call__(
        self, conv: torch.Tensor, aid: torch.Tensor, possible_actions: torch.Tensor, temperature: float = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(conv, aid, possible_actions, temperature)

    def forward(
        self, conv: torch.Tensor, aid: torch.Tensor, possible_actions: torch.Tensor, temperature: float = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(conv.unsqueeze(1)))  # Batch x 8 x 5 x 5

        x = F.relu(self.conv2(x))  # Batch x 16 x 3 x 3
        x = F.relu(self.conv3(x))  # Batch x 32 x 1 x 1

        x = F.relu(self.conv_dense(x.flatten(1)))  # Batch x 32

        x = torch.cat((x, aid), dim=1)  # Batch x (32 + AID_SHAPE[1])

        x = F.relu(self.dense1(x))  # Batch x 16
        x = self.dense2(x)  # Batch x 4

        x[:, 1:] = x[:, 1:].masked_fill(torch.logical_not(possible_actions), float("-inf"))

        return torch.sigmoid(x[:, 0]), F.softmax(x[:, 1:] / temperature, dim=1)


# class SnakeUNet(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#         self.conv_down1 = nn.Conv2d(1, 8, 3)  # 8x5x5
#         self.conv_down2 = nn.Conv2d(8, 16, 3)  # 16x3x3
#         self.conv_down3 = nn.Conv2d(16, 32, 3)  # 32x1x1

#         self.dense = nn.Linear(32 * (2 * VIEW_SIZE - 5) * (2 * VIEW_SIZE - 5), 32)

#         self.conv_up1 = nn.Conv2d(32 + AID_SHAPE[1], 16, 3, padding="same")
#         self.conv_up2 = nn.Conv2d(16, 8, 3, padding="same")
#         self.conv_up3 = nn.Conv2d(8, 4, 3, padding="same")

#     def __call__(
#         self, conv: torch.Tensor, aid: torch.Tensor, possible_actions: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         return super().__call__(conv, aid, possible_actions)

#     def forward(
#         self, conv: torch.Tensor, aid: torch.Tensor, possible_actions: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         x = F.relu(self.conv_down1(conv.unsqueeze(1)))  # Batch x 8 x 5 x 5

#         x = F.relu(self.conv_down2(x))  # Batch x 16 x 3 x 3
#         x = F.relu(self.conv_down3(x))  # Batch x 32 x 1 x 1

#         x = F.relu(self.dense(x.flatten(1)))  # Batch x 32

#         x = torch.cat((x, aid), dim=1)  # Batch x (32 + AID_SHAPE[1])
#         x = x.view(-1, 32 + AID_SHAPE[1], 1, 1)

#         x = F.pad(x, (1, 1))  # Batch x (32 + AID_SHAPE[1]) x 3 x 3
#         x = F.relu(self.conv_up1(x))  # Batch x 16 x 3 x 3
#         x = F.pad(x, (1, 1))  # Batch x 16 x 5 x 5
#         x = F.relu(self.conv_up2(x))  # Batch x 8 x 5 x 5
#         x = F.pad(x, (1, 1))  # Batch x 8 x 7 x 7
#         x = self.conv_up3(x)  # Batch x 4 x 7 x 7

#         actions = x[:, 0]
#         policies = F.softmax(x[:, 1:], dim=1)


class HumanEval(Evaluation):
    def __call__(
        self, conv: torch.Tensor, aid: torch.Tensor, possible_actions: torch.Tensor, temperature: float = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        len_0 = aid[0][0].item()
        len_1 = aid[1][0].item()

        value = (len_0 - len_1) / (11 * 11)
        values = torch.tensor([value, -value], dtype=torch.float, device=DEVICE)

        return values, F.softmax(aid[:, 1:] / temperature, dim=1)


class SnakeNetwork(Network):
    """
    Takes two snakes and evaluate who is winning

    Parameters:
        - (N_SNAKES, *CONV_SHAPE) for the view of the snakes
        - (N_SNAKES, *AID_SHAPE) for the helpful parameters of the snakes

    Returns 2 tensors of shape:
        - (1,) for who will win (1 for main player, -1 for other player, 0 for draw)
        - (N_SNAKES, N_ACTION - 1) values for the snakes action probabilities (cannot go backward because the conv tensor is rotated so that the snake is always going up)
    """

    def __init__(self, snake_net: SnakeNet | None = None) -> None:
        super().__init__()

        if snake_net is None:
            snake_net = SnakeNet().to(DEVICE)

        self.snake_net: SnakeNet = snake_net

    def forward(
        self, convs: torch.Tensor, aids: torch.Tensor, possible_actions: torch.Tensor, temperature: float = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, policies = self.snake_net(convs, aids, possible_actions, temperature)

        return values - values.mean(), policies
