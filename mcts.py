import numpy as np
import random
from constants import *
from copy import deepcopy
from sys import maxsize
from scipy.special import softmax

dh_group = [(None, None), ((np.rot90, 1), None), ((np.rot90, 2), None),
            ((np.rot90, 3), None), (np.fliplr, None), (np.flipud, None),
            (np.flipud,  (np.rot90, 1)), (np.fliplr, (np.rot90, 1))]

def sample_rotation(state, num=8):
    """ Apply a certain number of random transformation to the input state """
    random.shuffle(dh_group)
    states = []
    boards = (HISTORY + 1) * 2 ## Number of planes to rotate
    for idx in range(num):
        new_state = np.zeros((boards + 1, BOARD_SIZE, BOARD_SIZE,))
        new_state[:boards] = state[:boards]
        for grp in dh_group[idx]:
            for i in range(boards):
                if isinstance(grp, tuple):
                    new_state[i] = grp[0](new_state[i], k=grp[1])
                elif grp is not None:
                    new_state[i] = grp(new_state[i])

        new_state[boards] = state[boards]
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
		self.children = []
		self.parent = parent
		self.move = move

	def update(self, v):
		self.n += 1
		self.w = self.w + v
		self.q = self.w / self.n if self.n > 0 else 0

	def isLeaf(self):
		return len(self.children) == 0

	def getU(self):
		total = np.sum([sibling.n for sibling in self.parent.children])
		return C_PUCT * self.p * np.sqrt(total) / (1 + self.n)

	def expand(self, p):
		for i in range(p.shape[0]):
			if p[i] > 0:
				self.children.append(Node(parent=self, move=i, prob=p[i]))

class MCTS():
	def __init__(self):
		self.root = Node()
		self.numMoves = 0

	def play(self, board, player, competitive=False):
		# Run another 1600 sims
		self.runSims(board, player)
		# Find move
		move, p = None, None
		action_scores = np.array([child.n for child in self.root.children])
		total = np.sum(action_scores)
		p = action_scores / total	
		if competitive or self.numMoves >= 30:
			moves = np.array([child.move for child in self.root.children])
			moves = moves[np.where(action_scores == np.max(action_scores))[0]]
			move = np.random.choice(moves)	
		else:
			moves = [child.move for child in self.root.children]
			move = np.random.choice(moves, p=p)
		self.numMoves += 1
		# Advance MCTS tree
		for child in self.root.children:
			if child.move == move:
				self.root = child
				break
		return move, p

	def runSims(self, board, player):
		for i in range(MCTS_SIMS):
			boardCopy = deepcopy(board)
			current_node = self.root
			done = False; depth = 0
			while not current_node.isLeaf() and not done:
				child = self.select(current_node)
				boardCopy.step(child.move)
				# boardCopy.render()
				# print("Move = ", child.move)
				# print("Depth = ", depth)
				# depth += 1
				# input()
				current_node = child
			v = self.expandAndEval(current_node, boardCopy, player)
			self.backup(current_node, v)

	def select(self, node):
		# select best child as per UCT algo (if multiple best select randomly any)
		scores = [child.q + child.getU() for child in node.children]
		bestChildren = np.where(scores == np.max(scores))[0]
		bestChild = node.children[np.random.choice(bestChildren)]
		return bestChild

	def expandAndEval(self, node, board, player):
		# expand as per NN then backup value
		feature = player.feature(getState(sample_rotation(board.state, num=1)))
		p = player.policy(feature)
		v = player.value(feature)
		# print("Policy\n", p)
		p = constrainMoves(board, p[0].cpu().data.numpy())
		# print("Constrained policy\n", p)
		node.expand(p)
		return v[0].cpu().data.numpy()

	def backup(self, node, v):
		current_node = node
		while current_node.parent != None:
			current_node.update(v)
			current_node = current_node.parent