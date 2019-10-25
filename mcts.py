import numpy as np
import random
from constants import *
from copy import deepcopy

dh_group = [(None, None), ((np.rot90, 1), None), ((np.rot90, 2), None),
            ((np.rot90, 3), None), (np.fliplr, None), (np.flipud, None),
            (np.flipud,  (np.rot90, 1)), (np.fliplr, (np.rot90, 1))]

def sample_rotation(state, num=8):
    """ Apply a certain number of random transformation to the input state """
    random.shuffle(dh_group)
    states = []
    boards = (HISTORY + 1) * 2 ## Number of planes to rotate
    for idx in range(num):
        new_state = np.zeros((boards + 1, GOBAN_SIZE, GOBAN_SIZE,))
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
		return C_PUCT * self.prob * np.sqrt(total) / (1 + self.n)

	def expand(self, p):
		for i in range(p.shape[0]):
			if p[i] > 0:
				self.children.append(parent=self, move=i, prob=p[i])

class MCTS():
	def __init__(self):
		self.root = Node()
		self.numMoves = 0

	def play(self, board, player, competitive=False):
		# Run another 1600 sims
		runSims(board, player)
		# Find move
		move, p = None, None
		action_scores = [child.n for child in self.node.children]
		total = np.sum(action_scores)
        p = action_scores / total	
		if competitive or self.Moves >= 30:
            moves = np.where(action_scores == np.max(action_scores))[0]
            move = np.random.choice(moves)	
        else:
			move = np.random.choice(action_scores.shape[0], p=p)
		self.moves += 1
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
			done = False
			while not current_node.isLeaf() or not done:
				child = self.select(current_node)
				boardCopy.step(child.move)
				current_node = child
			v = expandAndEval(current_node, boardCopy, player)
			backup(current_node, v)

	def select(self, node):
		# select best child as per UCT algo
		scores = [child.q + child.getU() for child in node.children]
		bestChild = node.children[np.argmax(scores)]
		return bestChild

	def expandAndEval(self, node, board, player):
		# expand as per NN then backup value
		feature = player.feature(sample_rotation(state, num=1))
		p = player.policy(feature)
		v = player.value(feature)
		p = constrainMoves(board, p)
		node.expand(p)
		return v

	def backup(self, node, v)
		current_node = node
		while current_node.parent != None:
			current_node.update(v)
			current_node = current_node.parent