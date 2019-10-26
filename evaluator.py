from game import Game
from constants import *

evaluator = Game(None, mctsEnable=True)

def evaluate(best_model, model):
	wins = 0
	evaluator.player = model
	evaluator.opponent = best_model
	for i in range(EVAL_GAMES):
		wins += evaluator.play(opFirst= i < EVAL_GAMES/2)
	return (wins >= EVAL_THRESH * EVAL_GAMES)
