import time
import os
from game import Game
from agent import Player
from constants import *
import pandas as pd
from tqdm import tqdm

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