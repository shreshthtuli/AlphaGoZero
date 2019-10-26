import torch
from game import Game
from constants import *

evaluator = Game(None, mctsEnable=True)

def evaluateAndSave(model):
	best_model = torch.load(BEST_PATH)
	wins = 0
	evaluator.player = model
	evaluator.opponent = best_model
	for i in range(EVAL_GAMES):
		wins += evaluator.play(opFirst= i < EVAL_GAMES/2)
	updateModel = wins >= EVAL_THRESH * EVAL_GAMES
	if updateModel:
		torch.save(model, BEST_PATH)
		return model
	return best_model
