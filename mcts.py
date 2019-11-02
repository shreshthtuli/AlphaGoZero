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

def constrainMoves(board, p):
	legal_moves = board.get_legal_moves()
	check = np.ones(BOARD_SIZE ** 2 + 1)
	np.put(check, legal_moves, [0])
	check = check * (-maxsize - 1)
	newP = softmax(p + check)
	newP[np.where(check != 0)] = 0
	return newP

def getState(state):
	x = torch.from_numpy(np.array([state]))
	x = torch.tensor(x, dtype=torch.float, device=DEVICE)
	return x

class Node:
	def __init__(self, parent=None, prob=None, move=None):
		self.p = prob # probability of coming to this node
		self.n = 0 # number of visits to this node
		self.w = 0 # total action value
		self.q = 0 # mean action value
		self.sqrtTotal = 0.0 # sqrt of sum of n of children
		self.children = []
		self.parent = parent
		self.move = move

	def update(self, v):
		self.parent.sqrtTotal = sqrt((self.parent.sqrtTotal ** 2) + 1)
		self.n += 1
		self.w = self.w + v
		self.q = self.w / self.n

	def isLeaf(self):
		return len(self.children) == 0

	def getU(self):
		return C_PUCT * self.p * self.parent.sqrtTotal / (1 + self.n)

	def expand(self, p):
		for i in range(p.shape[0]):
			if p[i] > 0:
				self.children.append(Node(parent=self, move=i, prob=p[i]))

class MCTS():
	def __init__(self):
		self.root = Node()
		self.numMoves = 0

	def play(self, board, player, competitive=False, move_no=0):
		# Run another 1600 sims
		self.runSims(board, player, move_no)
		# Find move
		move, p = None, None
		action_scores = np.array([child.n for child in self.root.children])
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
		for child in self.root.children:
			if child.move == move:
				self.root = child
				break
		return move, return_scores

	def runSims(self, board, player, move_no=0):
		selectTime, expandTime, backupTime = 0, 0, 0
		totalTime, uctTime, whereTime, randomTime = 0, 0, 0, 0
		for i in range(MCTS_SIMS):
			t1 = time.time()
			boardCopy = deepcopy(board)
			current_node = self.root
			done = False; depth = 0
			while not current_node.isLeaf() and not done:
				child, totalTimeI, uctTimeI, whereTimeI, randomTimeI = self.select(current_node)
				totalTime += totalTimeI
				uctTime += uctTimeI
				whereTime += whereTimeI
				randomTime += randomTimeI
				boardCopy.step(child.move)
				# print(boardCopy.state)
				# boardCopy.render()
				# print("Move = ", child.move)
				# print("Depth = ", depth)
				# depth += 1
				# input()
				current_node = child
			t2 = time.time()
			v = self.expandAndEval(current_node, boardCopy, player)
			# print(boardCopy.state)
			t3 = time.time()
			# print("Backup value: ", v)
			self.backup(current_node, v)
			t4 = time.time()
			selectTime += t2-t1
			expandTime += t3-t2
			backupTime += t4-t3
		# print(move_no, selectTime, expandTime, backupTime)
		print(move_no, totalTime, uctTime, whereTime, randomTime)

	def select(self, node):
		# select best child as per UCT algo (if multiple best select randomly any)
		t1 = time.time()
		total = np.sum([child.n for child in node.children])
		t2 = time.time()
		scores = [child.q + child.getU() for child in node.children]
		t3 = time.time()
		bestChildren = np.where(scores == np.max(scores))[0]
		t4 = time.time()
		bestChild = node.children[np.random.choice(bestChildren)]
		t5 = time.time()
		return bestChild, t2-t1, t3-t2, t4-t3, t5-t4

	def expandAndEval(self, node, board, player):
		# expand as per NN then backup value
		feature = player.feature(getState(sample_rotation(board.state)))
		p = player.policy(feature)
		v = player.value(feature)
		# print("Policy\n", p)
		# print("Constrained policy\n", p)
		node.expand(constrainMoves(board, p[0].cpu().data.numpy()))
		return v[0].cpu().data.numpy()

	def backup(self, node, v):
		current_node = node
		while current_node.parent != None:
			current_node.update(v)
			current_node = current_node.parent
