import torch
import multiprocessing
nmc = multiprocessing.cpu_count()

# Number of cores
NUM_CORES = 1

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

# Kernel Size
KERNEL_SIZE = 3

C_PUCT = 0.2

# Path of best model
BEST_PATH = "bestModel.pth"

# Threshold to overwrite best player
EVAL_THRESH = 0.5

if NUM_CORES < 10:
	# Number of filters
	FILTERS = 32 # 256

	# Number of Residual Blocks
	BLOCKS = 8 #19

	# Number of games in self play
	GAMES = 1 * NUM_CORES # 25000
	TOTAL_GAMES = 50000 # 500k

	# Number of MCTS simulations
	MCTS_SIMS = 20 # 1600

	# milestones for changing learning rate
	MILESTONES = [400, 600] # 400, 600

	# Evaluation Games
	EVAL_GAMES = 2 * NUM_CORES # 400

	# batch size for training of policy+value network
	BATCH_SIZE_TRAIN = 64 # 2048 an 32 per worker

	# number of batches
	N_BATCHES = 500 # 1000
else:
	# Number of filters
	FILTERS = 256 # 256

	# Number of Residual Blocks
	BLOCKS = 13 #19

	# Number of games in self play
	GAMES = 2 * NUM_CORES # 25000
	TOTAL_GAMES = 50000 # 500k

	# Number of MCTS simulations
	MCTS_SIMS = 20 # 1600

	# milestones for changing learning rate
	MILESTONES = [400, 600] # 400, 600

	# Evaluation Games
	EVAL_GAMES = 1 * NUM_CORES # 400

	# batch size for training of policy+value network
	BATCH_SIZE_TRAIN = 128 # 2048 an 32 per worker

	# number of batches
	N_BATCHES = 1000 # 1000

