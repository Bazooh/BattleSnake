import os

import numpy as np
from tqdm import tqdm

from Utils.utils import *
from MCTS.Game import *

import torch
import torch.optim as optim

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
})

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OthelloNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = tuple(game.getBoardSize())
        self.action_size = game.getActionSize()
        self.args = args

        super(OthelloNNet, self).__init__()
        self.conv1 = nn.Conv2d(2, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4) + 4, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, grid, apples, heads):
        x = torch.stack((grid, apples), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))                          # batch_size, num_channels, board_x, board_y
        x = F.relu(self.bn2(self.conv2(x)))                          # batch_size, num_channels, board_x, board_y
        x = F.relu(self.bn3(self.conv3(x)))                          # batch_size, num_channels, (board_x-2), (board_y-2)
        x = F.relu(self.bn4(self.conv4(x)))                          # batch_size, num_channels, (board_x-4), (board_y-4)
        x = x.flatten(start_dim=1)
        x = torch.concat((x, heads), dim=1)

        x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p=self.args.dropout, training=self.training)  # batch_size, 1024
        x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p=self.args.dropout, training=self.training)  # batch_size, 512

        pi = self.fc3(x)                                                                         # batch_size, action_size
        v = self.fc4(x)                                                                          # batch_size, 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

class NNetWrapper():
    def __init__(self, game: Game):
        self.nnet = OthelloNNet(game, args)
        self.action_size = game.getActionSize()
        self.grid_bounds = game.getBoardSize()
    
    def board_to_tensor(self, board):
        grid = get_grid({**board[NODE_MAIN_SNAKE], **board[NODE_OTHER_SNAKES]}, self.grid_bounds)
        apples = np.zeros(self.grid_bounds)
        for apple in board[NODE_APPLES]:
            apples[apple[X], apple[Y]] = 1.
        return (
            torch.FloatTensor(grid).view((self.grid_bounds[X], self.grid_bounds[Y])),
            torch.FloatTensor(apples).view((self.grid_bounds[X], self.grid_bounds[Y])),
            torch.FloatTensor(np.concatenate((list(board[NODE_MAIN_SNAKE].values())[0][0], list(board[NODE_OTHER_SNAKES].values())[0][0])))
        )

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print(f'EPOCH ::: {epoch + 1}')
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """

        # preparing input
        grid, apples, heads = self.board_to_tensor(board)
        self.nnet.eval()
        
        with torch.no_grad():
            pi, v = self.nnet(
                grid.view(1, self.grid_bounds[X],self.grid_bounds[Y]),
                apples.view(1, self.grid_bounds[X],self.grid_bounds[Y]),
                heads.view(1, -1)
            )
        
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
