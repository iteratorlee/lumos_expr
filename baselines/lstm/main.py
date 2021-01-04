# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os, json

def getfeatures():
	features = {}
	Path = '/Users/macpro/Desktop/lumos-dataset/'
	vendors = ['alibaba/1 second/', 'huawei/1 second/', 'tencent/1 second/', 'aws/']
	vendors2 = ['alibaba/5 second/', 'huawei/5 second/', 'tencent/5 second/'] 
	for vendor in vendors:
		for instance in os.listdir(Path + vendor):
			if instance[0] == '.':
				continue
			path = Path + vendor + instance
			for bench in os.listdir(path):
				if bench[0] == '.' or bench[-1] in ['4', '5', '6']:
					continue
				with open(path + '/' + bench + '/report.json') as f:
					result = json.load(f)
					if not bench in features:
						features[bench] = {instance: [float(result['elapsed_time'])]}
					else:
						features[bench][instance] = [float(result['elapsed_time'])]
				with open(path + '/' + bench + '/sar.csv') as f:
					lines = f.readlines()[1:]
					for i in range(len(lines)):
						lines[i] = lines[i].strip().split(',')[1:]
						lines[i].remove('lo')
						for j in range(len(lines[i])):
							lines[i][j] = float(lines[i][j])
					features[bench][instance].append(lines)
	for vendor in vendors2:
		for instance in os.listdir(Path + vendor):
			if instance[0] == '.':
				continue
			path = Path + vendor + instance
			for bench in os.listdir(path):
				if bench[0] == '.' or bench[-1] in ['4', '5', '6'] or (('hadoop' in bench or 'spark' in bench) and ('tiny' in bench or 'small' in bench)):
					continue
				with open(path + '/' + bench + '/report.json') as f:
					result = json.load(f)
					if not bench in features:
						features[bench] = {instance: [float(result['elapsed_time'])]}
					else:
						features[bench][instance] = [float(result['elapsed_time'])]
				with open(path + '/' + bench + '/sar.csv') as f:
					lines = f.readlines()[1:]
					for i in range(len(lines)):
						lines[i] = lines[i].strip().split(',')[1:]
						lines[i].remove('lo')
						for j in range(len(lines[i])):
							lines[i][j] = float(lines[i][j])
					features[bench][instance].append(lines)
	return features

'''
features = getfeatures()
with open('features', 'w') as f:
	json.dump(features, f)
'''
with open('features') as f:
	features = json.load(f)

with open('detail_conf.json') as f:
	confs = json.load(f)

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
INPUT_SIZE = 61     # rnn input size
LR = 0.02           # learning rate

class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()

		self.rnn = nn.LSTM(
			input_size=INPUT_SIZE,
			hidden_size=32,     # rnn hidden unit
			num_layers=1,       # number of rnn layer
			batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
		)
		self.hidden1 = nn.Linear(32, 1)
		self.hidden2 = nn.Linear(6, 1)
		self.out = nn.Linear(4, 1)

	def forward(self, x, x2, x3, x4, x5):
		# x shape (batch, time_step, input_size)
		# r_out shape (batch, time_step, output_size)
		# h_n shape (n_layers, batch, hidden_size)
		# h_c shape (n_layers, batch, hidden_size)
		r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
		# choose r_out at the last time step
		tmp1 = self.hidden1(r_out[:, -1, :])
		tmp2 = self.hidden2(torch.cat([x2, x3], dim=1))
		out = self.out(torch.cat([tmp1, tmp2, x4, x5], dim=1))
		return out

rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

i = 0
for bench, benchdata in features.items():
	if not 'small' in bench or '_2' in bench or '_3' in bench:
		continue 
	benchlarge = bench.replace('small', 'large')
	benchhuge = bench.replace('small', 'huge')
	i += 1
	j = 0
	for instance, data in benchdata.items():
		x = torch.from_numpy(np.array(data[1], dtype=np.float32)).view(1, -1, INPUT_SIZE)    # shape (batch, time_step, input_size)
		x2 = torch.from_numpy(np.array(list(confs[instance].values())[:3], dtype=np.float32)).view(1, 3)
		x5 = torch.tensor([[data[0]]])
		print(i, j)
		j += 1
		for instance2, data2 in benchdata.items():
			y = torch.tensor([[data2[0]]])
			x3 = torch.from_numpy(np.array(list(confs[instance2].values())[:3], dtype=np.float32)).view(1, 3)
			x4 = torch.tensor([[1.0]])

			prediction = rnn(x, x2, x3, x4, x5)   # rnn output
			loss = loss_func(prediction, y)         # calculate loss
			optimizer.zero_grad()                   # clear gradients for this training step
			loss.backward()                         # backpropagation, compute gradients
			optimizer.step()                        # apply gradients

		if benchlarge in features:
			for instance2, data2 in features[benchlarge].items():
				y = torch.tensor([[data2[0]]])
				x3 = torch.from_numpy(np.array(list(confs[instance2].values())[:3], dtype=np.float32)).view(1, 3)
				x4 = torch.tensor([[2.0]])
				
				prediction = rnn(x, x2, x3, x4, x5)   # rnn output
				loss = loss_func(prediction, y)         # calculate loss
				optimizer.zero_grad()                   # clear gradients for this training step
				loss.backward()                         # backpropagation, compute gradients
				optimizer.step()                        # apply gradients
		
		if benchhuge in features:
			for instance2, data2 in features[benchhuge].items():
				y = torch.tensor([[data2[0]]])
				x3 = torch.from_numpy(np.array(list(confs[instance2].values())[:3], dtype=np.float32)).view(1, 3)
				x4 = torch.tensor([[3.0]])
				
				prediction = rnn(x, x2, x3, x4, x5)   # rnn output
				loss = loss_func(prediction, y)         # calculate loss
				optimizer.zero_grad()                   # clear gradients for this training step
				loss.backward()                         # backpropagation, compute gradients
				optimizer.step()                        # apply gradients

torch.save(rnn, 'rnn.pkl') 

# rnn = torch.load('rnn.pkl')

instance = 'hfr6.2xlarge'
data = features['spark_logisticregression_small_1'][instance]
x = torch.from_numpy(np.array(data[1], dtype=np.float32)).view(1, -1, INPUT_SIZE)    # shape (batch, time_step, input_size)
x2 = torch.from_numpy(np.array(list(confs[instance].values())[:3], dtype=np.float32)).view(1, 3)
x5 = torch.tensor([[data[0]]])
for instance2, data2 in features['spark_logisticregression_large_1'].items():
	y = torch.tensor([[data2[0]]])
	x3 = torch.from_numpy(np.array(list(confs[instance2].values())[:3], dtype=np.float32)).view(1, 3)
	x4 = torch.tensor([[2.0]])
	prediction = rnn(x, x2, x3, x4, x5)   # rnn output
	print(instance, ':', prediction, y)
for instance2, data2 in features['spark_logisticregression_small_1'].items():
	y = torch.tensor([[data2[0]]])
	x3 = torch.from_numpy(np.array(list(confs[instance2].values())[:3], dtype=np.float32)).view(1, 3)
	x4 = torch.tensor([[1.0]])
	prediction = rnn(x, x2, x3, x4, x5)   # rnn output
	print(instance, ':', prediction, y)