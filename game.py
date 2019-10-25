import numpy as np
from constants import *
from go import GoEnv as Board
from sys import maxsize
from scipy.special import softmax
import pandas as pd

np.set_printoptions(threshold=maxsize)

class Game:
	def __init__(self, player, color='black', opponent=None):
		# Create new board
		self.board = Board(color, BOARD_SIZE)
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
		check = check * (-maxsize - 1)
		newP = softmax(p + check)
		newP[np.where(check != 0)] = 0
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
		action = self.move(self.board, p)
		state, reward, done = self.board.step(action)
		return state, reward, done, action

	def play(self, opFirst=False):
		done = False
		state = self.board.reset()
		# Black plays first
		self.player_color = (1 if opFirst else 2) if self.opponent else 2
		print("Player color", self.player_color)
		datasetStates = []
		datasetActions = []
		datasetDone = []
		datasetRewards = []
		comp = False; reward = None

		if opFirst:
			state, reward, done, action = self.playOnce(self.getState(state), \
                    self.opponent, self.player.passed, competitive=True)
		while not done:
			if self.opponent:
				state, reward, done, action = self.playOnce(self.getState(state), \
                    self.player, self.opponent.passed, competitive=True)
				state, reward, done, action = self.playOnce(self.getState(state), \
                    self.opponent, self.player.passed, competitive=True)
			else:
				state = self.getState(state)
				new_state, reward, done, action = self.playOnce(state, self.player, \
                    False)
				datasetStates.append(state.data.numpy())
				datasetActions.append(action)
				datasetDone.append(done)
				# Set rewards as if winner is white
				datasetRewards.append(1 if self.player_color == 1 else -1)
				self.swap()
				state = new_state
		# reward is 1 if white wins
		print("Winner", 'white' if self.board.get_winner() == 1 else 'black')
		datasetRewards = np.multiply(datasetRewards, -1 if reward != 1 else 1)

		df = pd.DataFrame({
			"States": datasetStates,
			"Actions": datasetActions,
			"Rewards": datasetRewards,
			"Done": datasetDone })
		return df
