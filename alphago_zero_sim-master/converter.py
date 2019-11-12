import numpy as np
from utils_5.constants import *
from utils_5.go import GoEnv as Board
import pandas as pd
from utils_5.mcts import MCTS
from sys import maxsize
import time 
from utils_5.agent import Player

model = torch.load(BEST_PATH)

torch.save(model.state_dict(), BEST_PATH)