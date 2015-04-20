import random, math

class Connection:

	weight = None
	deltaWeight = None

	def __init__(self, weight = None, deltaWeight = None):
		self.weight = random.random() if weight == None else weight
		self.deltaWeight = 0.0 if deltaWeight == None else deltaWeight

	def toJSON(self):
		return {'__class__': 'Connection', '__weight__': self.weight, '__deltaWeight__': self.deltaWeight}

	@classmethod
	def fromJSON(cls, JSON):
		if '__class__' in JSON:
			if JSON['__class__'] == 'Connection':
				return Connection(JSON['__weight__'], JSON['__deltaWeight__'])

class Neuron:

	eta = 0.10 # [0.0..1.0] overall net training rate
	alpha = 0.5 # [0.0..n] multiplier of last weight change (momentum)
	output = None
	outputWeights = None
	index = None
	layer = None
	gradient = None

	def __init__(self, numOutputs, index, layer, eta = None, alpha = None, output = None, gradient = None, outputWeights = None):
		self.index = index
		self.layer = layer
		self.eta = 0.10 if eta == None else eta
		self.alpha = 0.5 if alpha == None else alpha
		self.output = 0.0 if output == None else output
		self.gradient = 0.0 if gradient == None else gradient
		if outputWeights == None:
			self.outputWeights = list()
			for c in range(numOutputs):
				self.outputWeights.append(Connection())
		else:
			self.outputWeights = outputWeights

	@classmethod
	def transferFunction(cls, x):
		return math.tanh(x)

	@classmethod
	def transferFunctionDerivative(cls, x):
		return 1.0 - x * x

	def feedForward(self, prevLayer):
		sum = 0.0
		for n in range(len(prevLayer)):
			sum += prevLayer[n].output * prevLayer[n].outputWeights[self.index].weight
		self.output = Neuron.transferFunction(sum)

	def calcOutputGradients(self, targetVal):
		self.gradient = (targetVal - self.output) * Neuron.transferFunctionDerivative(self.output)

	def calcHiddenGradients(self, nextLayer):
		sum = 0.0
		for n in range(len(nextLayer) - 1):
			sum += self.outputWeights[n].weight * nextLayer[n].gradient
		self.gradient = sum * Neuron.transferFunctionDerivative(self.output)

	def updateInputWeights(self, prevLayer):
		for n in range(len(prevLayer)):
			neuron = prevLayer[n]
			oldDeltaWeight = neuron.outputWeights[self.index].deltaWeight;
			newDeltaWeight = self.eta * neuron.output * self.gradient + self.alpha * oldDeltaWeight;
			neuron.outputWeights[self.index].deltaWeight = newDeltaWeight
			neuron.outputWeights[self.index].weight += newDeltaWeight

	def toJSON(self):
		return {'__class__': 'Neuron', '__eta__': self.eta, '__alpha__': self.alpha, '__gradient__': self.gradient, '__index__': self.index, '__layer__': self.layer, '__output__': self.output, '__outputWeights__': [connection.toJSON() for connection in self.outputWeights]}

	@classmethod
	def fromJSON(cls, JSON):
		if '__class__' in JSON:
			if JSON['__class__'] == 'Neuron':
				index = JSON['__index__']
				layer = JSON['__layer__']
				eta = JSON['__eta__']
				alpha = JSON['__alpha__']
				output = JSON['__output__']
				gradient = JSON['__gradient__']
				outputWeights = [Connection.fromJSON(connection) for connection in JSON['__outputWeights__']]
				return Neuron(None, index, layer, eta, alpha, output, gradient, outputWeights)

class Net:

	layers = None
	netError = None
	averageError = None
	smoothingFactor = None

	def __init__(self, topology, netError = None, averageError = None, smoothingFactor = None, layers = None, eta = None, alpha = None):
		self.netError = 0.0 if netError == None else netError
		self.averageError = 0.0 if averageError == None else averageError
		self.smoothingFactor = 100.0 if smoothingFactor == None else smoothingFactor
		if layers == None:
			numLayers = len(topology)
			self.layers = list()
			for layerNum in range(numLayers):
				self.layers.append(list())
				numOutputs = 0 if layerNum + 1 == len(topology) else int(topology[layerNum + 1])
				for neuronNum in range(int(topology[layerNum]) + 1):
					self.layers[-1].append(Neuron(numOutputs, neuronNum, layerNum, eta, alpha))
				self.layers[-1][-1].output = 1
		else:
			self.layers = layers

	def feedForward(self, inputVals):
		assert len(inputVals) + 1 == len(self.layers[0])
		for i in range(len(inputVals)):
			self.layers[0][i].output = inputVals[i]
		for layerNum in range(1, len(self.layers)):
			prevLayer = self.layers[layerNum - 1]
			for n in range(len(self.layers[layerNum]) - 1):
				self.layers[layerNum][n].feedForward(prevLayer)

	def backPropagate(self, targetVals):
		outputLayer = self.layers[-1]
		self.netError = 0.0
		for n in range(len(outputLayer) - 1):
			self.netError += (targetVals[n] - outputLayer[n].output) * (targetVals[n] - outputLayer[n].output)
		self.netError = math.sqrt(self.netError / (len(outputLayer) - 1))
		self.averageError = (self.averageError * self.smoothingFactor + self.netError) / (self.smoothingFactor + 1.0);
		for n in range(len(outputLayer) - 1):
			outputLayer[n].calcOutputGradients(targetVals[n])
		for layerNum in reversed(range(1, len(self.layers) - 1)):
			hiddenLayer = self.layers[layerNum]
			nextLayer = self.layers[layerNum + 1]
			for n in range(len(hiddenLayer)):
				hiddenLayer[n].calcHiddenGradients(nextLayer)
		for layerNum in reversed(range(1, len(self.layers))):
			layer = self.layers[layerNum]
			prevLayer = self.layers[layerNum - 1]
			for n in range(len(layer) - 1):
				layer[n].updateInputWeights(prevLayer)

	def toJSON(self):
		return {'__class__': 'Net', '__netError__': self.netError, '__averageError__': self.averageError, '__smoothingFactor__': self.smoothingFactor, '__layers__': [[neuron.toJSON() for neuron in layer] for layer in self.layers]}

	@classmethod
	def fromJSON(cls, JSON):
		if '__class__' in JSON:
			if JSON['__class__'] == 'Net':
				netError = JSON['__netError__']
				averageError = JSON['__averageError__']
				smoothingFactor = JSON['__smoothingFactor__']
				layers = [[Neuron.fromJSON(neuron) for neuron in layer] for layer in JSON['__layers__']]
				return Net(None, netError, averageError, smoothingFactor, layers)
