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
FILTERS = 20

# Kernel Size
KERNEL_SIZE = 3

# Number of Residual Blocks
BLOCKS = 19

# Number of games in self play
GAMES = 1

# Number of MCTS simulations
MCTS_SIMS = 10
C_PUCT = 0.2