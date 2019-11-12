import sys, ast
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Press c to move back, v to move forward
Usage: python vis.py [filename]
"""

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
	nodes = nx.draw_networkx_nodes(G, pos=pos, node_color=clr, node_size=400)
	if nodes:
		nodes.set_edgecolor('k')
	
def draw_ith_row_df(fig, data, i):
	state = data.iloc[i][0]
	plot_grid(*board_shape)
	plot_stones_color(np.array(state[0]), 'k')
	plot_stones_color(np.array(state[8]), 'w')

def press(event):
	global curr_index, max_data, data, fig, ax
	if event.key == "c" and curr_index>0:
		curr_index-=1
	elif event.key == "v" and curr_index<max_data:
		curr_index+=1
	# print("pressed", event.key, curr_index)
	if event.key in ["c", "v"]:
		ax.cla()
		ax.set_facecolor('wheat')
		draw_ith_row_df(fig, data, curr_index)
		fig.canvas.draw()
	
BOARDSIZE = 13	
curr_index = 0
board_shape = (BOARDSIZE, BOARDSIZE)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_facecolor('wheat')

max_data = 0
fig.canvas.mpl_connect('key_press_event', press)

df = pd.read_pickle(sys.argv[1])
data = df["States"]
max_data = df.shape[0]-1
totaldata = len(data)
draw_ith_row_df(fig, data, curr_index)

plt.show()