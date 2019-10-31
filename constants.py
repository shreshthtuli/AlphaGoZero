import torch

# Size of Go Board
BOARD_SIZE = 13

# Device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Number of Games to bw considered in state
HISTORY = 7

# Input Planes
INPLANES = (HISTORY + 1) * 2 + 1

# Policy Output
POLICY_OUTPUT = BOARD_SIZE * BOARD_SIZE + 1

# Number of filters
FILTERS = 256 # 256

# Kernel Size
KERNEL_SIZE = 3

# Number of Residual Blocks
BLOCKS = 13 #19

# Number of games in self play
GAMES = 1 # 25000

# Number of MCTS simulations
MCTS_SIMS = 120 # 1600
C_PUCT = 0.2

# milestones for changing learning rate
MILESTONES = [40, 60] # 400, 600

# Evaluation Games
EVAL_GAMES = 3 # 400

# Threshold to overwrite best player
EVAL_THRESH = 0.55

# batch size for training of policy+value network
BATCH_SIZE_TRAIN = 32 # 2048 an 32 per worker

# number of batches
N_BATCHES = 100 # 1000

# Path of best model
BEST_PATH = "bestModel.pth"

