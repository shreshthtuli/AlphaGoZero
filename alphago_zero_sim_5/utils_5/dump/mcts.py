import numpy as np
import random
from constants import *
from copy import deepcopy
from sys import maxsize
from scipy.special import softmax
import time
import warnings
from math import sqrt
warnings.simplefilter("ignore")

all_moves = np.arange(BOARD_SIZE ** 2 + 1)

dh_group = [(None, None), ((np.rot90, 1), None), ((np.rot90, 2), None),
            ((np.rot90, 3), None), (np.fliplr, None), (np.flipud, None),
            (np.flipud,  (np.rot90, 1)), (np.fliplr, (np.rot90, 1))]

def sample_rotation(state):
    """ Apply a certain number of random transformation to the input state """
    states = []
    boards = (HISTORY + 1) * 2 ## Number of planes to rotate
    dh = dh_group[np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])]
    new_state = np.copy(state)
    for grp in dh:
        for i in range(boards):
            if isinstance(grp, tuple):
                new_state[i] = grp[0](new_state[i], k=grp[1])
            elif grp is not None:
                new_state[i] = grp(new_state[i])
    states.append(new_state)
    if len(states) == 1:
        return np.array(states[0])
    return np.array(states)

def constrainMoves(board, p, moveno=100):
	legal_moves = board.get_legal_moves()
	illegal_moves = np.setdiff1d(all_moves,np.array(legal_moves))
	p[illegal_moves] = 0
	if moveno < NOPASS_MULTPLR * MOVE_LIMIT:
		p[-1] = 0
	total = np.sum(p)
	p /= total
	return p

def getState(states):
	x = torch.from_numpy(np.array(states))
	x = torch.tensor(x, dtype=torch.float, device=DEVICE)
	return x

def dirichlet_noise(probas):
    dim = (probas.shape[0],)
    new_probas = (1 - EPS) * probas + \
                    EPS * np.random.dirichlet(np.full(dim, ALPHA))
    return new_probas

class Node:
	def __init__(self, parent=None, prob=None, move=None):
		self.p = prob # probability of coming to this node
		self.n = 0 # number of visits to this node
		self.w = 0 # total action value
		self.q = 0 # mean action value
		self.sqrtTotal = 0.0 # sqrt of sum of n of children
		self.children = []
		self.parent = parent
		self.qPlusU = self.getU() if self.parent else 0
		self.move = move
		self.bestChild = None

	def update(self, v):
		self.parent.sqrtTotal = sqrt((self.parent.sqrtTotal ** 2) + 1)
		self.n += 1
		self.w = self.w + v
		self.q = self.w / self.n
		oldqu = self.qPlusU
		self.qPlusU = self.q + self.getU()
		if self.parent and self.qPlusU < oldqu:
			self.parent.findBest()

	def isLeaf(self):
		return len(self.children) == 0

	def getU(self):
		return C_PUCT * self.p * self.parent.sqrtTotal / (1 + self.n)

	def expand(self, p):
		for i in range(p.shape[0]):
			if p[i] > 0:
				self.children.append(Node(parent=self, move=i, prob=p[i]))
		self.findBest()

	def findBest(self):
		scores = [child.qPlusU for child in self.children]
		bestChildren = np.where(scores == np.max(scores))[0]
		bestChild = self.children[np.random.choice(bestChildren)]
		self.bestChild = bestChild

class MCTS():
	def __init__(self):
		self.root = Node()
		self.numMoves = 0

	def play(self, board, player, competitive=False, moveno=100):
		# 1 step lookahead evaluation
		self.TuliSharmaOptimization(player, board, competitive)
		# Run another 1600 sims
		self.runSims(board, player, moveno)
		# Find move
		move, p = None, None
		action_scores = np.array([child.n-1 for child in self.root.children])
		return_scores = np.zeros(BOARD_SIZE ** 2 + 1)
		total = np.sum(action_scores)
		p = action_scores / total	
		if competitive or self.numMoves >= 30:
			moves = np.array([child.move for child in self.root.children])
			bestmoves = moves[np.where(action_scores == np.max(action_scores))[0]]
			move = np.random.choice(bestmoves)	
		else:
			moves = [child.move for child in self.root.children]
			move = np.random.choice(moves, p=p)
		self.numMoves += 1
		return_scores[moves] = p
		# Advance MCTS tree
		self.advance(move)
		return move, return_scores

	def advance(self, move):
		for child in self.root.children:
			if child.move == move:
				self.root = child
				break

	def runSims(self, board, player, moveno=100):
		#selectTime, expandTime, backupTime = 0, 0, 0
		startTime = time.time()
		for i in range(MCTS_SIMS):
			if time.time() - startTime > 2:
				break
			# t1 = time.time()
			boardCopy = deepcopy(board)
			current_node = self.root
			done = False; depth = 0; v = 0
			while not current_node.isLeaf() and not done:
				child = self.select(current_node)
				_, winner, done = boardCopy.step(child.move)
				# print(boardCopy.state)
				# boardCopy.render()
				# print("Move = ", child.move)
				# print("Depth = ", depth)
				depth += 1
				# input()
				current_node = child
			# t2 = time.time()
			if done:
				v = 1 if winner == board.player_color else -1
			else:
				v = self.expandAndEval(current_node, boardCopy, player, moveno+depth)
			#t3 = time.time()
			self.backup(current_node, v)
		# 	t4 = time.time()
		# 	selectTime += t2-t1
		# 	expandTime += t3-t2
		# 	backupTime += t4-t3
		# print(selectTime, expandTime, backupTime)

	def select(self, node):
		# select best child as per UCT algo (if multiple best select randomly any)
		return node.bestChild

	def expandAndEval(self, node, board, player, moveno=100):
		# expand as per NN then backup value
		feature = player.feature(getState([sample_rotation(board.state)]))
		p = player.policy(feature)
		v = player.value(feature)
		node.expand(constrainMoves(board, p[0].cpu().data.numpy(), moveno))
		return v[0].cpu().data.numpy()

	def backup(self, node, v):
		current_node = node
		while current_node.parent != None:
			current_node.update(v)
			current_node = current_node.parent

	def TuliSharmaOptimization(self, player, board, competitive):
		feature = player.feature(getState([sample_rotation(board.state)]))
		p = player.policy(feature)
		# Expand root to explore all 1 step moves
		if len(self.root.children) == 0:
			if competitive:
				self.root.expand(constrainMoves(board, p[0].cpu().data.numpy()))
			else:
				self.root.expand(constrainMoves(board, dirichlet_noise(p[0].cpu().data.numpy()))) 
		# Evaluate all 1 step moves
		childStates = []
		for child in self.root.children:
			boardCopy = deepcopy(board)
			boardCopy.step(child.move)
			childStates.append(sample_rotation(boardCopy.state))
		features = player.feature(getState(childStates))
		v = player.value(features).cpu().data.numpy()
		# Update q values of these states
		for i in range(len(self.root.children)):
			self.root.children[i].update(v[i][0])
