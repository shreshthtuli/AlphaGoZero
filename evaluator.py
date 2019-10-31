import torch
from game import Game
from constants import *
from joblib import Parallel, delayed
import numpy as np

evaluators = [Game(None, mctsEnable=True) for c in range(NUM_CORES)]

def getRes(evaluator, opf):
	return evaluator.play(opFirst = opf)

def evaluateAndSave(model):
	best_model = torch.load(BEST_PATH).to(DEVICE)
	wins = 0
	for evaluator in evaluators:
		evaluator.player = model
		evaluator.opponent = best_model
	results = Parallel(n_jobs=NUM_CORES)(delayed(getRes)(evaluators[i], i<EVAL_GAMES/2) for i in range(NUM_CORES))
	print(results)
	wins = np.sum(results)
	updateModel = wins >= EVAL_THRESH * EVAL_GAMES
	if updateModel:
		torch.save(model, BEST_PATH)
		return model
	return best_model
