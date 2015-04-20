#!/usr/bin/python
import os, sys, json
from DNN import Connection, Neuron, Net
myMet = None
inputs = None
outputs = None
count = 0
error = 0.0
with open(os.environ['NNJSON'], mode = 'r') as f:
	myNet = Net.fromJSON(json.load(f))
for line in sys.stdin:
	tokens = line.strip().split()
	numInputs = int(tokens[0])
	numOutputs = int(tokens[1])
	if len(tokens) == 2 + numInputs + numOutputs and numInputs + 1 == len(myNet.layers[0]) and numOutputs + 1 == len(myNet.layers[-1]):
		inputs = [float(token) for token in tokens[2 : 2 + numInputs]]
		outputs = [float(token) for token in tokens[2 + numInputs: 2 + numInputs + numOutputs]]
		myNet.feedForward(inputs)
		for index in range(len(myNet.layers[-1]) - 1):
			error += (outputs[index] - myNet.layers[-1][index].outputVal) * (outputs[index] - myNet.layers[-1][index].outputVal)
	count += 1
print count, error
