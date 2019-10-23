import numpy as numpy
from constants import *
from go import GoEnv as Board
from sys import maxint
from scipy.special import softmax
import pandas as pd

class Game:
	def __init__(self, player, opponent, color):
		# Create new board
		self.board = Board(color, BOARD_SIZE)
		self.board.reset()
		self.player_color = 2 if color == "black" else 1
		self.player = player
		self.opponent = opponent

	def reset(self):
		self.board.reset()

	def swap(self):
		self.player_color = (self.player_color % 2) + 1

	def move(self, board, p):
		legal_moves = board.get_legal_moves()
		check = np.ones(BOARD_SIZE ** 2 + 1)
		np.put(check, legal_moves, [0])
		check = check * (-maxint - 1)
		newP = softmax(p + check)
		move = np.random.choice(newP.shape[0], p=newP)
		return move

	def getState(self, state):
		x = torch.from_numpy(np.array([state]))
		x = torch.tensor(x, dtype=torch.float, device=DEVICE)
		return x

	def playOnce(self, state, player, other_pass):
		feature = player.feature(state)
		p = player.policy(feature)
		p = p[0].cpu().data.numpy()
		action = self.move(state, p)
		state, reward, done = self.board.step(action)
		return state, reward, done, action


	def play(self):
		done = False
		state = self.board.reset()
		datasetStates = []
		datasetActions = []
		datasetDone = []
		datasetRewards = []

		while not done:
			if self.opponent:
				state, reward, done, action = self.playOnce(self.getState(state), \
                    self.player, self.opponent.passed, competitive=True)
				state, reward, done, action = self.playOnce(self.getState(state), \
                    self.player, self.opponent.passed, competitive=True)
			else:
				state = self.getState(state)
				new_state, reward, done, action = self._play(state, self.player, \
                    False, competitive=comp)
				self.swap()
				datasetStates.append(state)
				datasetActions.append(action)
				datasetRewards.append(reward)
				datasetDone.append(done)

		df = pd.DataFrame({
			"States": datasetStates,
			"Actions": datasetActions,
			"Rewards": datasetRewards,
			"Done": datasetDone })
		return df



