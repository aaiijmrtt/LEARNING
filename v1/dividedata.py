#!/usr/bin/python
import os, sys, json
from DNN import Connection, Neuron, Net
if not os.path.exists(sys.argv[4]):
	eta = float(sys.argv[5])
	alpha = float(sys.argv[6])
	topology = [int(number) for number in sys.argv[7:]]
	myNet = Net(topology = topology, eta = eta, alpha = alpha)
	with open(sys.argv[4], mode = 'w') as f:
		json.dump(myNet.toJSON(), f, indent = 4)
lines = open(sys.argv[1], 'r').readlines()
divider = int(sys.argv[3])
test = float(sys.argv[2]) * len(lines)
count = 0
f = None
for line in lines:
	if count % divider == 0:
		if not f == None:
			f.close()
		if count < test:
			f = open('testdata/'+str(int(count/divider))+'.in', 'w')
		else:
			f = open('traindata/'+str(int(count/divider))+'.in', 'w')
	f.write(line)
	count += 1
if not f == None:
	f.close()
