from torch.utils.data.dataset import Dataset

class Position_Sampler(Dataset):
	def __init__(self, df):
		self.df = df
		print(self.df["ActionScores"].iloc[2300])
		
	def __len__(self):
		return self.df.shape[0]
		
	def __getitem__(self, idx):
		data = dict()
		data['states'] = self.df["States"].iloc[idx][0]
		data['vals'] = self.df["Rewards"].iloc[idx]
		data['probs'] = self.df["ActionScores"].iloc[idx]
		return data