from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import torch
from torch.autograd import Variable

from constants import *

def make_best_global_model(model):
	pass

def cross_entropy_mod(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim = 1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
	
def evaluate(model):
	return 0.5

def train(train_loader, model):
	model.train()
	
	epoch = 0
	
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
	scheduler = MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)
	
	criterion1 = nn.MSELoss()
	
	iter = 0
	for i, batch_data in enumerate(train_loader):
		optimizer.zero_grad()
		
		states = batch_data['states'].float()
		true_vals = batch_data['vals'].float()
		true_probs = batch_data['probs'].float()
		
		## preprocessing here if needed depends on data loader
		####
		
		states = Variable(states).to(DEVICE)
		true_vals = Variable(true_vals).to(DEVICE)
		true_probs = Variable(true_probs).to(DEVICE)
		
		f_map = model.feature(states)
		pred_vals = model.value(f_map)
		pred_probs = model.policy(f_map)
		pred_vals = torch.squeeze(pred_vals)
		mse_loss = criterion1(pred_vals, true_vals)
		cross_entropy_loss = cross_entropy_mod(pred_probs, true_probs)
		loss = 0.5*mse_loss + 0.5*cross_entropy_loss
		loss.backward()
		
		optimizer.step()
		
		iter += 1
		# print(iter)
		
		if iter == N_BATCHES:
			return model
			
		scheduler.step()
		