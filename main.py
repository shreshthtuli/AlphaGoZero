import time
import os
import pandas as pd
from tqdm import tqdm

from agent import Player
from constants import *
from train import *
from sys import platform
if platform == 'linux':
	from evaluator import *
	from game import Game
	simulator = Game(alphazero, mctsEnable=True)
from data import *

print(DEVICE)

alphazero = Player().cuda()

while True:
	# New dataset
	dataset = pd.DataFrame({
				"States": [],
				"Actions": [],
				"ActionScores": [],
				"Rewards": [],
				"Done": []})
	# Geneate dataset by self play
	if platform == 'linux':
		for i in tqdm(range(GAMES)):
			df = simulator.play()
			dataset = dataset.append(df)

	# dataset.to_pickle('dataset.pkl')
	# dataset = pd.read_pickle('dataset.pkl')

	# Train player
	train_data = Position_Sampler(dataset)
	data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE_TRAIN)
	train(data_loader, alphazero)
	# Evaluate player
	evaluateAndSave(alphazero)