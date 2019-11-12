import numpy as np
from utils.constants import *
from utils.go import GoEnv as Board
import pandas as pd
from utils.mcts import MCTS
from sys import maxsize
import time 
from utils.agent import Player

model = torch.load(BEST_PATH)

torch.save(model.state_dict(), BEST_PATH)