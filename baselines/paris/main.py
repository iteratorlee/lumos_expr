# -*- coding: utf-8 -*-

import numpy as np  
import json, os
from sklearn.ensemble import RandomForestRegressor  

features = {}
confs = {}
instances = []
Path = '/Users/macpro/Desktop/lumos-dataset/'
vendors = ['alibaba/1 second/', 'huawei/1 second/', 'tencent/1 second/', 'aws/']
vendors2 = ['alibaba/5 second/', 'huawei/5 second/', 'tencent/5 second/'] 
for vendor in vendors:
	for instance in os.listdir(Path + vendor):
		if instance[0] == '.':
			continue
		path = Path + vendor + instance
		instances.append(instance)
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
with open('detail_conf.json') as f:
	confs = json.load(f)

X = []
y = []
benches = {}
cost = []

for bench, datas in features.items():
	if bench[-2:] == '_2' or bench[-2:] == '_3':# or not 'large' in bench:
		continue
	for instance, data in datas.items():
		tmp = list(confs[instance].values())[0:3]
		tmp.extend(datas['hfr6.2xlarge'])
		tmp.extend(datas['s6.2xlarge.2'])
		# {'g6.large': 223, 'r6.2xlarge': 225, 'g6.2xlarge': 226, 'hfc6.large': 217, 'hfg6.xlarge': 223, 'hfr6.xlarge': 223, 'hfg6.2xlarge': 225, 'c6.xlarge': 223, 'hfg6.large': 223, 'hfr6.2xlarge': 226, 'c6.large': 217, 'r6.large': 223, 'g6.xlarge': 223, 'hfc6.2xlarge': 226, 'r6.xlarge': 223, 'hfr6.large': 223, 'c6.2xlarge': 226, 'hfc6.xlarge': 223, 'm6.large.8': 223, 'm6.2xlarge.8': 226, 'c6.large.2': 217, 'c6.2xlarge.2': 226, 'c6.2xlarge.4': 225, 'c6.large.4': 223, 's6.xlarge.4': 218, 's6.xlarge.2': 220, 'c6.xlarge.4': 216, 'c6.xlarge.2': 221, 's6.large.4': 214, 's6.2xlarge.4': 226, 's6.large.2': 217, 's6.2xlarge.2': 226, 'm6.xlarge.8': 223, 'c3.large8': 223, 'c3.2xlarge32': 226, 's5.large16': 197, 's5.medium8': 198, 'm5.2xlarge64': 226, 'm5.large32': 223, 'c3.large16': 223, 'c3.2xlarge16': 225, 's5.2xlarge16': 225, 's5.large8': 198, 'm5.medium16': 223, 's5.2xlarge32': 201, 's5.medium4': 192, 'm5.large': 212, 'c5.large': 196, 'r5.xlarge': 210, 'r5.2xlarge': 213, 'm5.2xlarge': 208, 'm5.xlarge': 210, 'c5.2xlarge': 211, 'r5.large': 211, 'c5.xlarge': 211}
		X.append(tmp)
		y.append(data[0]/datas['hfr6.2xlarge'][0])
		cost.append(list(confs[instance].values())[3])
		if bench[-2:] == '_1':
			if not bench[:-2] in benches:
				benches[bench[:-2]] = []
			benches[bench[:-2]].append(len(y)-1)
		else:
			if not bench in benches:
				benches[bench] = []
			benches[bench].append(len(y)-1)

total = 0
correct1 = 0	
correct2 = 0
print(len(instances))
print(len(list(benches.keys())))

t = 0
for bench, indexes in benches.items():
	t += 1
	print(t, bench)
	train_X = X[:]
	train_y = y[:]
	test_X = []
	test_y = []
	tmpcost = []
	for i in sorted(indexes, reverse=True):
		del train_X[i]
		del train_y[i]
	for i in indexes:
		test_X.append(X[i])
		test_y.append(y[i])
		tmpcost.append(cost[i])
	rf = RandomForestRegressor(n_estimators=10, random_state=0) 
	rf.fit(train_X, train_y)
	predict_y = rf.predict(test_X)
	print(test_y, predict_y, tmpcost)
	'''
	min_index = 0
	min_result = 9999
	for k in range(len(predict_y)):
		if predict_y[k] < min_result:
			min_result = predict_y[k]
			min_index = k
	result = test_y[min_index]
	count = 0
	flag1 = True
	flag2 = True
	for k in test_y:
		if k < result:
			count += 1
		# threshold
		if count > 1:
			flag1 = False
		if count > 3:
			flag2 = False
			break
	'''
	min_index = 0
	min_result = 9999
	for k in range(len(predict_y)):
		if predict_y[k]*tmpcost[k] < min_result:
			min_result = predict_y[k]*tmpcost[k]
			min_index = k
	result = test_y[min_index]*tmpcost[min_index]
	count = 0
	flag1 = True
	flag2 = True
	for i, k in enumerate(test_y):
		if k*tmpcost[i] < result:
			count += 1
		# threshold
		if count > 1:
			flag1 = False
		if count > 3:
			flag2 = False
			break
	
	total += 1
	if flag1:
		correct1 += 1
	if flag2:
		correct2 += 1
print(correct1 / total)
print(correct2 / total)

'''
train_X = []
train_y = []
for bench, indexes in benches.items():
	if 'small' in bench:
		for i in indexes:
			train_X.append(X[i])
			train_y.append(y[i])
	
rf = RandomForestRegressor(n_estimators=10, random_state=0) 
rf.fit(train_X, train_y)

for bench, indexes in benches.items():
	test_X = []
	test_y = []
	tmpcost = []
	if not 'huge' in bench:
		continue
	for i in indexes:
		test_X.append(X[i])
		test_y.append(y[i])
		tmpcost.append(cost[i])
	predict_y = rf.predict(test_X)
	
	min_index = 0
	min_result = 9999
	for k in range(len(predict_y)):
		if predict_y[k] < min_result:
			min_result = predict_y[k]
			min_index = k
	result = test_y[min_index]
	count = 0
	flag1 = True
	flag2 = True
	for k in test_y:
		if k < result:
			count += 1
		# threshold
		if count >= 6:
			flag1 = False
		if count >= 12:
			flag2 = False
			break

	min_index = 0
	min_result = 9999
	for k in range(len(predict_y)):
		if predict_y[k]*tmpcost[k] < min_result:
			min_result = predict_y[k]*tmpcost[k]
			min_index = k
	result = test_y[min_index]*tmpcost[min_index]
	count = 0
	flag1 = True
	flag2 = True
	for i, k in enumerate(test_y):
		if k*tmpcost[i] < result:
			count += 1
		# threshold
		if count >= 6:
			flag1 = False
		if count >= 12:
			flag2 = False
			break

	total += 1
	if flag1:
		correct1 += 1
	if flag2:
		correct2 += 1
print(correct1 / total)
print(correct2 / total)
'''
