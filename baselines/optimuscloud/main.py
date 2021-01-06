# -*- coding: utf-8 -*-

import numpy as np  
import json, os
from sklearn.ensemble import RandomForestRegressor  

# 'small', 'large' or 'all'
train_scale = 'small'
predict_scale = 'large'
top = 5
normalize_rate = 30
# choose 2 instance types as reference, use runtime on them to predict runtime on other VMs
reference_instances = ['hfr6.2xlarge', 's6.2xlarge.2']
n_estimators = 10

def get_features():
	"""
	Get performance metrics of all workloads on all VMs
	Change Path to lumos-dataset
	"""
	features = {}
	confs = {}
	with open('detail_conf.json') as f:
		confs = json.load(f)
	Path = '/Users/macpro/Desktop/lumos-dataset/'
	vendors = ['alibaba/1 second/', 'huawei/1 second/', 'tencent/1 second/', 'aws/']
	vendors2 = ['alibaba/5 second/', 'huawei/5 second/', 'tencent/5 second/'] 
	for vendor in vendors:
		for instance in os.listdir(Path + vendor):
			if instance[0] == '.':
				continue
			path = Path + vendor + instance
			for bench in os.listdir(path):
				if bench[-2:] == '_2' or bench[-2:] == '_3':
					continue
				if bench[0] == '.' or bench[-1] in ['4', '5', '6']:
					continue
				with open(path + '/' + bench + '/report.json') as f:
					result = json.load(f)
					if not bench in features:
						features[bench] = {instance: [float(result['elapsed_time'])]}
					else:
						features[bench][instance] = [float(result['elapsed_time'])]
				with open(path + '/' + bench + '/sar.csv') as f:
					lines = f.readlines()[1:-1]
					l = len(lines)
					tmp = lines[int(l/4)].strip().split(',')[1:]
					tmp.remove('lo')
					tmp.extend(lines[int(2*l/4)].strip().split(',')[1:])
					tmp.remove('lo')
					tmp.extend(lines[int(3*l/4)].strip().split(',')[1:])
					tmp.remove('lo')
					features[bench][instance].extend(tmp)
	for vendor in vendors2:
		for instance in os.listdir(Path + vendor):
			if instance[0] == '.':
				continue
			path = Path + vendor + instance
			for bench in os.listdir(path):
				if bench[-2:] == '_2' or bench[-2:] == '_3':
					continue
				if bench[0] == '.' or bench[-1] in ['4', '5', '6'] or (('hadoop' in bench or 'spark' in bench) and ('tiny' in bench or 'small' in bench)):
					continue
				with open(path + '/' + bench + '/report.json') as f:
					result = json.load(f)
					if not bench in features:
						features[bench] = {instance: [float(result['elapsed_time'])]}
					else:
						features[bench][instance] = [float(result['elapsed_time'])]
				with open(path + '/' + bench + '/sar.csv') as f:
					lines = f.readlines()[1:-1]
					l = len(lines)
					tmp = lines[int(l/4)].strip().split(',')[1:]
					tmp.remove('lo')
					tmp.extend(lines[int(2*l/4)].strip().split(',')[1:])
					tmp.remove('lo')
					tmp.extend(lines[int(3*l/4)].strip().split(',')[1:])
					tmp.remove('lo')
					features[bench][instance].extend(tmp)
	return features, confs

def handle_features(features, confs):
	"""
	get train data X, y from original features
	return costs: cost of instance of i-th train data 
	       benches: benchname of i-th train data 
	"""
	X = []
	y = []
	costs = []
	benches = []
	for bench, datas in features.items():
		for instance, data in datas.items():
			# target instance information
			tmp = list(confs[instance].values())[0:3]
			# performance metrics on two reference VMs
			tmp.extend(datas[reference_instances[0]])
			tmp.extend(datas[reference_instances[1]])
			X.append(tmp)
			# normalized time
			y.append(data[0]*normalize_rate/datas[reference_instances[0]][0])
			costs.append(list(confs[instance].values())[3])
			benches.append(bench)
	return X, y, costs, benches

def get_accuracy(train_X, train_y, test_X, test_y, benches, costs, top, same=False):
	"""
	calculate jct accuracy and cost accuracy
	top: top n
	same: whether the train_data and predict_data are the same, 
		if true, extract the using one predict_data from train_data
		else, append two datas on reference VMs from predict_data to train_data 
	"""
	total = 0
	jct_correct = 0	
	cost_correct = 0
	rf = RandomForestRegressor(n_estimators=n_estimators, random_state=0) 
	if not same:
		rf.fit([t[1] for t in train_X], [t[1] for t in train_y])
	all_bench = set([benches[t[0]] for t in test_y])
	for bench in all_bench:
		if same:
			rf.fit([t[1] for t in train_X if benches[t[0]] != bench], [t[1] for t in train_y if benches[t[0]] != bench])
		predict_X = [t[1] for t in test_X if benches[t[0]] == bench]
		real_y = [t[1] for t in test_y if benches[t[0]] == bench]
		tmpcosts = [costs[t[0]] for t in test_y if benches[t[0]] == bench]
		predict_y = rf.predict(predict_X)
		
		chooseninstance = np.argmin(np.array(predict_y))
		result = real_y[chooseninstance]

		count = 0
		flag = True
		for time in real_y:
			if time < result:
				count += 1
			# threshold
			if count > top:
				flag = False
				break
		if flag:
			jct_correct += 1

		chooseninstance = np.argmin(np.array(predict_y)*np.array(tmpcosts))
		result = real_y[chooseninstance] * tmpcosts[chooseninstance]

		count = 0
		flag = True
		for j, time in enumerate(real_y):
			if time * tmpcosts[j] < result:
				count += 1
			# threshold
			if count > top:
				flag = False
				break
		if flag:
			cost_correct += 1

		total += 1
	return jct_correct / total, cost_correct / total

if __name__ == '__main__':
	features, confs = get_features()
	X, y, costs, benches = handle_features(features, confs)
	train_X = []; train_y = []
	test_X = []; test_y = []
	if train_scale == 'all':
		train_X = [(i,t) for i, t in enumerate(X) if 'small' in benches[i] or 'large' in benches[i]]
		train_y = [(i,t) for i, t in enumerate(y) if 'small' in benches[i] or 'large' in benches[i]]
	else:
		train_X = [(i,t) for i, t in enumerate(X) if train_scale in benches[i]]
		train_y = [(i,t) for i, t in enumerate(y) if train_scale in benches[i]]
	if predict_scale == 'all':
		test_X = [(i,t) for i, t in enumerate(X) if 'small' in benches[i] or 'large' in benches[i]]
		test_y = [(i,t) for i, t in enumerate(y) if 'small' in benches[i] or 'large' in benches[i]]
	else:
		test_X = [(i,t) for i, t in enumerate(X) if predict_scale in benches[i]]
		test_y = [(i,t) for i, t in enumerate(y) if predict_scale in benches[i]]
	if train_scale != predict_scale:
		jct_accuracy, cost_accuracy = get_accuracy(train_X, train_y, test_X, test_y, benches, costs, top)
	else:
		jct_accuracy, cost_accuracy = get_accuracy(train_X, train_y, test_X, test_y, benches, costs, top, True)
	print(jct_accuracy, cost_accuracy)

"""
Results
small -> large
top1: 0.43, 0.93; top3: 0.60, 0.97; top5: 0.70, 0.97
all -> all
top1: 0.48, 0.92; top3: 0.64, 0.95; top5: 0.79, 0.97
large -> large
top1: 0.50, 0.93; top3: 0.63, 0.97; top5: 0.80, 0.97
"""