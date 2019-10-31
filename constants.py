import torch
import multiprocessing
num_cores = multiprocessing.cpu_count()

# Number of cores
NUM_CORES = num_cores

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
EVAL_THRESH = 0.55

if NUM_CORES < 10:
	# Number of filters
	FILTERS = 16 # 256

	# Number of Residual Blocks
	BLOCKS = 3 #19

	# Number of games in self play
	GAMES = 4 # 25000
	TOTAL_GAMES = 20 * GAMES # 500k

	# Number of MCTS simulations
	MCTS_SIMS = 10 # 1600

	# milestones for changing learning rate
	MILESTONES = [40, 60] # 400, 600

	# Evaluation Games
	EVAL_GAMES = 4 # 400

	# batch size for training of policy+value network
	BATCH_SIZE_TRAIN = 32 # 2048 an 32 per worker

	# number of batches
	N_BATCHES = 100 # 1000
else:
	# Number of filters
	FILTERS = 256 # 256

	# Number of Residual Blocks
	BLOCKS = 13 #19

	# Number of games in self play
	GAMES = 80 # 25000
	TOTAL_GAMES = 20 * GAMES # 500k

	# Number of MCTS simulations
	MCTS_SIMS = 20 # 1600

	# milestones for changing learning rate
	MILESTONES = [400, 600] # 400, 600

	# Evaluation Games
	EVAL_GAMES = 40 # 400

	# batch size for training of policy+value network
	BATCH_SIZE_TRAIN = 32 # 2048 an 32 per worker

	# number of batches
	N_BATCHES = 1000 # 1000

