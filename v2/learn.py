#!/usr/bin/python

# usage: python learn.py [datafile: string] [fractiontest: float] [parallel: int] [modelfile: string] [crossover: float] [mutation: float] [generations: int] [extinction:float] [eta: float] [alpha: float] [topology: list(int)]

import sys, os, json, threading, shutil, random
from NN import Connection, Neuron, Net
from GA import NeuralGene, Generation

def divide_data(datafile, fractiontest, parallel, modelfile, eta, alpha, topology):
	lines = open(datafile, 'r').readlines()
	test = int(fractiontest * len(lines))
	train = int((len(lines) - test) / parallel)
	os.makedirs('testdata')
	os.makedirs('traindata')
	f = open('testdata/test.in', mode = 'w')
	random.shuffle(lines)
	for line in lines[0: test]:
		f.write(line)
	f.close()
	for i in range(parallel):
		if not os.path.exists(modelfile+str(i)+'.json'):
			myNet = Net(topology = topology, eta = eta, alpha = alpha)
			with open(modelfile+str(i)+'.json', mode = 'w') as f:
				json.dump(myNet.toJSON(), f, indent = 4)
		f = open('traindata/train'+str(i)+'.in', mode = 'w')
		for line in lines[test + i * train: test + (i + 1) * train]:
			f.write(line)
		f.close()

def train_net(datafile, modelfile):
	myNet = None
	with open(modelfile, mode = 'r') as f:
		myNet = Net.fromJSON(json.load(f))
	for line in open(datafile, mode = 'r').readlines():
		tokens = line.strip().split()
		numInputs = int(tokens[0])
		numOutputs = int(tokens[1])
		if len(tokens) == 2 + numInputs + numOutputs and numInputs + 1 == len(myNet.layers[0]) and numOutputs + 1 == len(myNet.layers[-1]):
			inputs = [float(token) for token in tokens[2 : 2 + numInputs]]
			outputs = [float(token) for token in tokens[2 + numInputs: 2 + numInputs + numOutputs]]
			myNet.feedForward(inputs)
			myNet.backPropagate(outputs)
	with open(modelfile, mode = 'w') as f:
		json.dump(myNet.toJSON(), f, indent = 4)

def train_data(parallel, modelfile):
	models = [modelfile+str(i)+'.json' for i in range(parallel)]
	random.shuffle(models)
	threads = [threading.Thread(name='thread'+str(i), target=train_net, args=('traindata/train'+str(i)+'.in', models[i])) for i in range(parallel)]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

def evolve_nets(parallel, modelfile, crossover, mutation, extinction, generations):
	nets = list()
	population = list()
	for i in range(parallel):
		with open(modelfile+str(i)+'.json', mode = 'r') as f:
			nets.append(Net.fromJSON(json.load(f)))
		population.append(NeuralGene(nets[-1], 'testdata/test.in'))
	generation = Generation(crossover, mutation, extinction, population)
	for i in range(generations):
		generation.evolve(parallel)
	for i in range(parallel):
		with open(modelfile+str(i)+'.json', mode = 'w') as f:
			json.dump(nets[i].toJSON(), f, indent = 4)
	return [g.fitness() for g in generation.population]

def delete_data():
	shutil.rmtree('testdata')
	shutil.rmtree('traindata')

if __name__ == '__main__':
	datafile = sys.argv[1]
	fractiontest = float(sys.argv[2])
	parallel = int(sys.argv[3])
	modelfile = sys.argv[4]
	crossover = float(sys.argv[5])
	mutation = float(sys.argv[6])
	extinction = float(sys.argv[7])
	generations = int(sys.argv[8])
	eta = float(sys.argv[9])
	alpha = float(sys.argv[10])
	topology = [int(number) for number in sys.argv[11:]]
	print '[CHECK] DIVIDING DATA'
	divide_data(datafile, fractiontest, parallel, modelfile, eta, alpha, topology)
	print '[CHECK] TRAINING DATA'
	train_data(parallel, modelfile)
	print '[CHECK] EVOLVING NETS'
	evolve_nets(parallel, modelfile, crossover, mutation, extinction, generations)
	print '[CHECK] DELETING DATA'
	delete_data()
