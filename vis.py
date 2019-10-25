import sys, ast
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_grid(rows, cols):
	G = nx.grid_2d_graph(rows, cols)
	pos = dict((n,n) for n in G.nodes())
	nx.draw_networkx(G, pos=pos, node_size=0, with_labels=False)
		
def plot_stones_color(locs, clr):
	rows, cols = locs.shape
	G = nx.grid_2d_graph(rows, cols)
	for i in range(rows):
		for j in range(cols):
			if locs[i][j] == 0:
				G.remove_node((i,j))
	pos = dict((n,n) for n in G.nodes())
	nodes = nx.draw_networkx_nodes(G, pos=pos, node_color=clr, node_size=460)
	nodes.set_edgecolor('k')
	
def draw_ith_row_df(df, i, board_shape):
	data = df["States"].iloc[i]
	data = ast.literal_eval(data)
	print(data)
	
	# plot_stones_color(np.array(data[0]), 'k')
	# plot_stones_color(np.array(data[8]), 'w')
	
BOARDSIZE = 13	
board_shape = (BOARDSIZE, BOARDSIZE)


fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_facecolor('wheat')
plot_grid(*board_shape)

df = pd.read_csv(sys.argv[1])
draw_ith_row_df(df, 0, board_shape)

# plt.show()