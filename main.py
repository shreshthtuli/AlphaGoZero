import time
import os
from game import Game
from agent import Player
from constants import *
import pandas as pd
from tqdm import tqdm

alphazero = Player()
simulator = Game(alphazero, None)

dataset = pd.DataFrame({
			"States": [],
			"Actions": [],
			"Rewards": [],
			"Done": []})

for i in tqdm(range(GAMES)):
	color = "white" if i < GAMES/2 else "black"
	df = simulator.play(color)

	dataset.append(df, ignore_index=True)

dataset.to_csv("data.csv")