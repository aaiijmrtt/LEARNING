#!/usr/bin/python
import os, sys, json
from DNN import Connection, Neuron, Net
myNet = None
inputs = None
outputs = None
with open(os.environ['NNJSON'], mode = 'r') as f:
	myNet = Net.fromJSON(json.load(f))
assert not myNet == None
for line in sys.stdin:
	tokens = line.strip().split()
	netLayer = int(tokens[0])
	neuronIndex = int(tokens[1])
	connectionIndex = int(tokens[2])
	deltaWeight = float(tokens[3])
	myNet.layers[netLayer][neuronIndex].outputWeights[connectionIndex].weight += deltaWeight
with open(os.environ['NNJSON'], mode = 'w') as f:
	json.dump(myNet.toJSON(), f, indent = 4)
