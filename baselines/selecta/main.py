# -*- coding: utf-8 -*-

import os, json
import collections
import joblib
import numpy as np
import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate

def get_times():
	result_times = {}
	instances = []
	Path = '/Users/macpro/Desktop/scout-CH/'
	vendors = ['alibaba/', 'huawei/', 'tencent/'] 
	for vendor in vendors:
		for instance in os.listdir(Path + vendor):
			if instance[0] == '.':
				continue
			path = Path + vendor + instance
			instances.append(vendor + instance)
			for bench in os.listdir(path):
				if bench[0] == '.' or bench[-1] in ['4', '5', '6']:
					continue
				with open(path + '/' + bench + '/report.json') as f:
					result = json.load(f)
					if not bench in result_times:
						result_times[bench] = {vendor+instance: float(result['elapsed_time'])}
					else:
						result_times[bench][vendor+instance] = float(result['elapsed_time'])
	return result_times, instances

def prepare_data(bench_times, instances, benches):
	with open('data.csv', 'w') as f:
		for bench, result in bench_times.items():
			for instance, time in result.items():
				f.write(str(benches.index(bench))+','+str(instances.index(instance))+','+str(time)+'\n')   
	with open('indexs.txt', 'w') as f:
		f.write(str(benches)+'\n')
		f.write(str(instances))

def normalize(p):
	bench_times = joblib.load('bench_times')
	with open('indexs.txt') as f:
		benches = eval(f.readline().strip())
		instances = eval(f.readline().strip())
	with open('normalized_data.csv', 'w') as f:
		for bench, result in bench_times.items():
			baseline = result['alibaba/hfr6.2xlarge'] / p
			for instance, time in result.items():
				time = time / baseline
				f.write(str(benches.index(bench))+','+str(instances.index(instance))+','+str(time)+'\n')



if __name__ == '__main__':
	'''
	bench_times, instances = get_times()
	joblib.dump(bench_times, 'bench_times')
	benches = [bench for bench in bench_times.keys()]
	prepare_data(bench_times, instances, benches)
	'''
	normalize(30)
	with open('indexs.txt') as f:
		benches = eval(f.readline().strip())
		instances = eval(f.readline().strip())

	instanceId = [str(instances.index('alibaba/hfr6.2xlarge')), str(instances.index('tencent/s5.large8'))]
	datas = []
	with open('normalized_data.csv') as f:
		for line in f.readlines():
			line = line.strip().split(',')
			datas.append(line)

	already_done = []
	correct = 0
	total = 0
	for i in range(len(datas)):
		bench_name = benches[int(datas[i][0])]
		'''
		# workload to be predicted
		if not 'tiny' in bench_name:
			continue
		'''
		if 'als_small' in bench_name or datas[i][0] in already_done:
			continue
		print(bench_name)
		samebench = []
		real_times = {}
		for t in datas:
			if bench_name[:-1] in benches[int(t[0])]:
				if t[0] in real_times:
					real_times[t[0]].append(float(t[2]))
				else:
					real_times[t[0]] = [float(t[2])]
				if t[0] not in samebench:
					samebench.append(t[0])
		print(samebench)
		already_done.extend(samebench)
		# small predict large
		# data = [t for t in datas if 'tiny' in benches[int(t[0])] and (not t[0] in samebench or t[1] in instanceId)]
		data = [t for t in datas if not t[0] in samebench or t[1] in instanceId]
		data = np.array(data)
		'''
		with open('temp.txt', 'w') as f:
			for t in data:
				f.write(str(t))
		'''
		ratings_dict = {'user': data[:, 0], 'item': data[:, 1], 'rating': data[:, 2]}
		df = pd.DataFrame(ratings_dict)

		reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 100))
		data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader=reader)
		trainset = data.build_full_trainset()
		svd = SVD()
		svd.fit(trainset)
		for t in samebench:
			predict_results = []
			for j in range(46):
				pred = svd.predict(t, str(j))
				predict_results.append(pred.est)
			min_index = 0
			min_result = 200
			for k in range(len(predict_results)):
				if predict_results[k] < min_result:
					min_result = predict_results[k]
					min_index = k
			result = real_times[t][min_index]
			count = 0
			flag = True
			for k in real_times[t]:
				if k < result:
					count += 1
				# threshold
				if count >= 4:
					flag = False
					break
			print(result, count)
			total += 1
			if flag:
				correct += 1
	print(correct / total)

