import time
import os
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib
from joblib import Parallel, delayed
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from agent import Player
from constants import *
from train import *
from sys import platform
from agent import Player
from data import *
import gc

num_cores = NUM_CORES

print(DEVICE, num_cores)
vHistory = []
pHistory = []
fig, ax1 = plt.subplots()
ax1.set_xlabel('iterations * 100')
ax1.set_ylabel('Value Loss', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax2 = ax1.twinx()
ax2.set_ylabel('Policy Loss', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
alphazero = Player().to(DEVICE)
torch.save(alphazero, BEST_PATH)
if platform == 'linux':
	from evaluator import *
	from game import Game
	simulators = [Game(alphazero, mctsEnable=True) for c in range(num_cores)]

def genGame(sim):
	localdf = pd.DataFrame({
				"States": [],
				"Actions": [],
				"ActionScores": [],
				"Rewards": [],
				"Done": []})	
	sim.player = alphazero
	for i in range(int(GAMES/num_cores)):
		df = sim.play()
		localdf = pd.concat([localdf, df])
	return localdf

startTime = time.time()
numLoops = 0
while True:
	dataset = pd.DataFrame({
				"States": [],
				"Actions": [],
				"ActionScores": [],
				"Rewards": [],
				"Done": []})

	# Generate dataset by self play
	if platform == 'linux':
		results = Parallel(n_jobs=num_cores)(delayed(genGame)(s) for s in simulators)
		results.append(dataset)
		# results = [genGame(simulators[0])]
		dataset = pd.concat(results)
		dataset = dataset[-1 * TOTAL_GAMES:]

	print("Epoch count: ", numLoops)
	print("time:", time.time() - startTime)
	startTime = time.time()
	
	# dataset.to_pickle('dataset.pkl')
	# dataset.to_csv('dataset.csv')
	# dataset = pd.read_pickle('dataset.pkl')

	# Train player
	train_data = Position_Sampler(dataset)

	sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(train_data) for i in range(len(train_data))],    \
																	 num_samples=BATCH_SIZE_TRAIN*N_BATCHES, replacement=True)
	# print(list(sample_strategy))
	data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE_TRAIN, sampler=sample_strategy)
	alphazero, vL, pL = train(data_loader, alphazero)
	print("Training complete")
	print("Value loss ", vL[-1], ", Policy loss ", pL[-1])
	vHistory.extend(vL); pHistory.extend(pL)
	ax1.cla(); ax2.cla()
	ax1.plot(range(len(vHistory)), vHistory, 'r')
	ax2.plot(range(len(vHistory)), pHistory, 'b')
	fig.tight_layout()
	fig.savefig("loss.pdf")
	# Evaluate player
	if numLoops > 15:
		MCTS_SIMS = min(50, 5 + numLoops)
		alphazero = evaluateAndSave(alphazero)
		print("Evaluation complete")
	numLoops += 1