import numpy as np
from utils.constants import *
from utils.go import GoEnv as Board
import pandas as pd
from utils.mcts import MCTS
from sys import maxsize
import time 
from agent import Player

class AlphaGoPlayer():
    def __init__(self, init_state, seed, player_color):
        self.board = Board("black", BOARD_SIZE)
        self.state = self.board.reset()
        self.player_color = player_color
        self.player = torch.load(BEST_PATH)
        self.done = False
        self.mcts = MCTS()

    def playOnce(self, other_pass):
        if other_pass and self.board.get_winner() + 1 == self.board.player_color:
            action = 169
        else: 
            action, action_scores = self.mcts.play(self.board, self.player, True)
        self.state, _, self.done = self.board.step(action)
        return action

    def get_action(self, cur_state, opponent_action):
        # print('--------------START--------------')
        if opponent_action != -1:
            # print("opponent_action ", opponent_action)
            self.mcts.advance(opponent_action)
            self.state, _, self.done = self.board.step(opponent_action)
            # self.board.render()
        if self.done:
            return 169 # pass
        action = self.playOnce(opponent_action == 169)
        # print("my action ", action)
        # self.board.render()
        # print('-------------- END --------------')
        return action
