import torch
import multiprocessing
nmc = multiprocessing.cpu_count()

# Number of cores
NUM_CORES = 1

# MCTS TIME
MCTS_TIME = 3

# Size of Go Board
BOARD_SIZE = 13

# Device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Epsilon for dirichlet
EPS = 0.25

# input to dirichlet funciton
ALPHA = 0.03

# Number of Games to bw considered in state
HISTORY = 7 

# Input Planes
INPLANES = (HISTORY + 1) * 2 + 1

# Policy Output
POLICY_OUTPUT = BOARD_SIZE * BOARD_SIZE + 1

# Kernel Size
KERNEL_SIZE = 3

# Move limit
MOVE_LIMIT = BOARD_SIZE * BOARD_SIZE

# How many moves without passing multiplier
NOPASS_MULTPLR = 0.3

C_PUCT = 0.2

# Path of best model
BEST_PATH = "./utils/bestModel.pth"

# Path of current model
CUR_PATH = "/scratch/cse/btech/cs1160311/curModel_200.pth"

# Threshold to overwrite best player
EVAL_THRESH = 0.5

if NUM_CORES < 6:
	# Number of filters
	FILTERS = 128 # 256

	# Number of Residual Blocks
	BLOCKS = 10 #19

	# Number of games in self play
	GAMES = 1 * NUM_CORES # 25000
	TOTAL_GAMES = 20000 # 500k

	# Number of MCTS simulations
	MCTS_SIMS = 1000 # 1600

	# milestones for changing learning rate
	MILESTONES = [200, 300] # 400, 600

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
	GAMES = 1 * NUM_CORES # 25000
	TOTAL_GAMES = 50000 # 500k

	# Number of MCTS simulations
	MCTS_SIMS = 200 # 1600

	# milestones for changing learning rate
	MILESTONES = [400, 600] # 400, 600

	# Evaluation Games
	EVAL_GAMES = 1 * NUM_CORES # 400

	# batch size for training of policy+value network
	BATCH_SIZE_TRAIN = 128 # 2048 an 32 per worker

	# number of batches
	N_BATCHES = 1000 # 1000

