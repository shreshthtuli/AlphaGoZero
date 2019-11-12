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
from os import path
from evaluator import *
from game import Game

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

if path.exists(BEST_PATH):
	alphazero = torch.load(BEST_PATH)
else:
	alphazero = Player().to(DEVICE)
	torch.save(alphazero, BEST_PATH)

simulators = [Game(alphazero, mctsEnable=True) for c in range(num_cores)]

def genGame(sim, movelimit):
	localdf = pd.DataFrame({
				"States": [],
				"Actions": [],
				"ActionScores": [],
				"Rewards": [],
				"Done": []})	
	sim.player = alphazero
	for i in range(int(GAMES/num_cores)):
		df = sim.play(movelimit=movelimit)
		localdf = pd.concat([localdf, df])
	return localdf

startTime = time.time()
numLoops = 0

if path.exists('dataset.pkl'):
	dataset = pd.read_pickle('dataset.pkl')
else:
	dataset = pd.DataFrame({
					"States": [],
					"Actions": [],
					"ActionScores": [],
					"Rewards": [],
					"Done": []})

multplr = 2
movelimit = MOVE_LIMIT*multplr
while True:
	# Generate dataset by self play
	results = Parallel(n_jobs=num_cores)(delayed(genGame)(s, movelimit) for s in simulators)
	# results = [genGame(simulators[0])]
	lenGames = [res.shape[0] for res in results]
	limitct = 0
	for lenGame in lenGames:
		if (lenGame*GAMES/num_cores) >= movelimit-1:
			limitct += 1
	if limitct > 0:
		multplr = min(2.5, 0.1+multplr)
		movelimit = multplr*MOVE_LIMIT
	
	print(lenGames)
	dataset = dataset.append(pd.concat(results, ignore_index=True), ignore_index=True)
	dataset = dataset[-1 * TOTAL_GAMES:]
	
	print("Epoch count: ", numLoops)
	print("Dataset size: ", dataset.shape[0])
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
	vHistory.extend(vL); pHistory.extend(pL)
	print("Value loss ", vHistory[-1], ", Policy loss ", pHistory[-1])
	ax1.cla(); ax2.cla()
	ax1.plot(range(len(vHistory)), vHistory, 'r')
	ax2.plot(range(len(vHistory)), pHistory, 'b')
	fig.tight_layout()
	fig.savefig("loss_200n5.pdf")
	# Evaluate player
	if numLoops > 5 and numLoops % 3 == 0:
	 	#alphazero = evaluateAndSave(alphazero)
	 	#print("Evaluation complete")
		torch.save(alphazero, BEST_PATH)
		dataset.to_pickle('dataset.pkl')
	numLoops += 1
