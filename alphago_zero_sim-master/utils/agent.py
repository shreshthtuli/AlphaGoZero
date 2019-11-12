import os
from .constants import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
	def __init__(self):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(FILTERS, FILTERS, kernel_size=3,
							stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(FILTERS)
		self.conv2 = nn.Conv2d(FILTERS, FILTERS, kernel_size=3,
							stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(FILTERS)

	def forward(self, x):
		residual = x
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += residual
		out = F.relu(out)
		return out

class Player(nn.Module):
    def __init__(self):
        super(Player, self).__init__()
        self.passed = False
        # Feature Extractor
        self.conv1 = nn.Conv2d(INPLANES, FILTERS, stride=1, 
							kernel_size=KERNEL_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(FILTERS)

        for block in range(BLOCKS):
            setattr(self, "res{}".format(block), \
				BasicBlock())
        # Policy Head
        self.convPolicy = nn.Conv2d(FILTERS, 2, kernel_size=1)
        self.bnPolicy = nn.BatchNorm2d(2)
        self.fc = nn.Linear(BOARD_SIZE * BOARD_SIZE * 2,
        					POLICY_OUTPUT)
        self.softmax = nn.Softmax(dim=1)
        # Value Head
        self.convValue = nn.Conv2d(FILTERS, 1, kernel_size=1)
        self.bnValue = nn.BatchNorm2d(1)
        self.fcValue1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 256)
        self.fcValue2 = nn.Linear(256, 1)

    def feature(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(BLOCKS - 1):
            x = getattr(self, "res{}".format(block))(x)

        feature_map = getattr(self, "res{}".format(BLOCKS - 1))(x)
       	return feature_map

    def policy(self, x):
    	x = F.relu(self.bnPolicy(self.convPolicy(x)))
    	x = x.view(-1, BOARD_SIZE * BOARD_SIZE * 2)
    	x = self.fc(x)
    	p = self.softmax(x)
    	return p

    def value(self, x):
    	x = F.relu(self.bnValue(self.convValue(x)))
    	x = x.view(-1, BOARD_SIZE * BOARD_SIZE)
    	x = F.relu(self.fcValue1(x))
    	v = torch.tanh(self.fcValue2(x))
    	return v
		
