import torch
from game import Game
from constants import *
from joblib import Parallel, delayed
import numpy as np

evaluators = [Game(None, mctsEnable=True) for c in range(NUM_CORES)]

def getRes(evaluator, opf):
	wins = 0
	for i in range(int(EVAL_GAMES/NUM_CORES)):
		wins += evaluator.play(opFirst = opf)
	return wins

def evaluateAndSave(model):
	best_model = torch.load(BEST_PATH).to(DEVICE)
	wins = 0
	for evaluator in evaluators:
		evaluator.player = model
		evaluator.opponent = best_model
	results = Parallel(n_jobs=NUM_CORES)(delayed(getRes)(evaluators[i], i<NUM_CORES/2) for i in range(NUM_CORES))
	print(results)
	wins = np.sum(results)
	updateModel = wins >= EVAL_THRESH * EVAL_GAMES
	if updateModel:
		torch.save(model, BEST_PATH)
		return model
	return best_model
