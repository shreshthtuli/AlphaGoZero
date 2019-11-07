import torch
from game import Game
from constants import *
from joblib import Parallel, delayed
import numpy as np
from sys import argv
from agent import Player

if argv[2] == 'manual':
	NUM_CORES = 1
	evaluators = [Game(None, mctsEnable=True, manual=True) for c in range(NUM_CORES)]
	EVAL_GAMES = 1
else:
	evaluators = [Game(None, mctsEnable=True) for c in range(NUM_CORES)]
	EVAL_GAMES = 20

def getRes(evaluator, opf):
	wins = 0
	for i in range(int(EVAL_GAMES/NUM_CORES)):
		wins += evaluator.play(opFirst = opf)
	return wins

def eval(model, othermodel):
	wins = 0
	for evaluator in evaluators:
		evaluator.player = model
		evaluator.opponent = othermodel
	results = Parallel(n_jobs=NUM_CORES)(delayed(getRes)(evaluators[i], i<NUM_CORES/2) for i in range(NUM_CORES))
	print(results)
	wins = np.sum(results)
	print(argv[1] + ' wins ' + str(wins) + ' games against ' + argv[2])


models = []

for a in argv[1:]:
	print(a)
	if a == 'manual':
		continue
	if a == 'random':
		models.append(Player().to(DEVICE))
	else:
		models.append(torch.load('bestModel-lr=' + a + '.pth'))

if argv[2] == 'manual':
	models.append(Player().to(DEVICE))
eval(models[0], models[1])