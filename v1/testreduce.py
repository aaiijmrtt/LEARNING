#!/usr/bin/python
import sys, os, json
from DNN import Connection, Neuron, Net
count = 0
error = 0.0
for line in sys.stdin:
	tokens = line.strip().split()
	count += int(tokens[0])
	error += float(tokens[1])
myNet = None
with open(os.environ['NNJSON'], mode = 'r') as f:
	myNet = Net.fromJSON(json.load(f))
if myNet.recentAverageError == 0:
	prevCount = 0
else:
	prevCount = int(myNet.error / myNet.recentAverageError)
myNet.error += error
myNet.recentAverageError = myNet.error / (count + prevCount)
with open(os.environ['NNJSON'], mode = 'w') as f:
	json.dump(myNet.toJSON(), f, indent = 4)
