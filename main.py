import time
import os
import pandas as pd
from tqdm import tqdm

from game import Game
from agent import Player
from constants import *
from train import *
# from data import *

alphazero = Player()
simulator = Game(alphazero)

dataset = pd.DataFrame({
			"States": [],
			"Actions": [],
			"ActionScores": [],
			"Rewards": [],
			"Done": []})

for i in tqdm(range(GAMES)):
	df = simulator.play()

	dataset = dataset.append(df)

dataset.to_pickle('dataset.pkl')

# train_data = dataset[:train_percentage*len(dataset)]
# data_loader = torch.utils.data.DataLoader(train_data, ...)
# train(data_loader, alphazero)