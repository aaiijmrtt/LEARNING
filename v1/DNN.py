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
	outputVal = None
	outputWeights = None
	myIndex = None
	myLayer = None
	gradient = None

	def __init__(self, numOutputs, myIndex, myLayer, eta = None, alpha = None, outputVal = None, gradient = None, outputWeights = None):
		self.myIndex = myIndex
		self.myLayer = myLayer
		self.eta = 0.10 if eta == None else eta
		self.alpha = 0.5 if alpha == None else alpha
		self.outputVal = 0.0 if outputVal == None else outputVal
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
			sum += prevLayer[n].outputVal * prevLayer[n].outputWeights[self.myIndex].weight
		self.outputVal = Neuron.transferFunction(sum)

	def calcOutputGradients(self, targetVal):
		self.gradient = (targetVal - self.outputVal) * Neuron.transferFunctionDerivative(self.outputVal)

	def calcHiddenGradients(self, nextLayer):
		sum = 0.0
		for n in range(len(nextLayer) - 1):
			sum += self.outputWeights[n].weight * nextLayer[n].gradient
		self.gradient = sum * Neuron.transferFunctionDerivative(self.outputVal)

	def updateInputWeights(self, prevLayer):
		for n in range(len(prevLayer)):
			neuron = prevLayer[n]
			oldDeltaWeight = neuron.outputWeights[self.myIndex].deltaWeight;
			newDeltaWeight = self.eta * neuron.outputVal * self.gradient + self.alpha * oldDeltaWeight;
			neuron.outputWeights[self.myIndex].deltaWeight = newDeltaWeight
			print neuron.myLayer, neuron.myIndex, self.myIndex, newDeltaWeight

	def toJSON(self):
		return {'__class__': 'Neuron', '__eta__': self.eta, '__alpha__': self.alpha, '__gradient__': self.gradient, '__myIndex__': self.myIndex, '__myLayer__': self.myLayer, '__outputVal__': self.outputVal, '__outputWeights__': [connection.toJSON() for connection in self.outputWeights]}

	@classmethod
	def fromJSON(cls, JSON):
		if '__class__' in JSON:
			if JSON['__class__'] == 'Neuron':
				myIndex = JSON['__myIndex__']
				myLayer = JSON['__myLayer__']
				eta = JSON['__eta__']
				alpha = JSON['__alpha__']
				outputVal = JSON['__outputVal__']
				gradient = JSON['__gradient__']
				outputWeights = [Connection.fromJSON(connection) for connection in JSON['__outputWeights__']]
				return Neuron(None, myIndex, myLayer, eta, alpha, outputVal, gradient, outputWeights)

class Net:

	layers = None
	error = None
	recentAverageError = None
	recentAverageSmoothingFactor = None # number of training samples to average over

	def __init__(self, topology, error = None, recentAverageError = None, recentAverageSmoothingFactor = None, layers = None, eta = None, alpha = None):
		self.error = 0.0 if error == None else error
		self.recentAverageError = 0.0 if recentAverageError == None else recentAverageError
		self.recentAverageSmoothingFactor = 100.0 if recentAverageSmoothingFactor == None else recentAverageSmoothingFactor
		if layers == None:
			numLayers = len(topology)
			self.layers = list()
			for layerNum in range(numLayers):
				self.layers.append(list())
				numOutputs = 0 if layerNum + 1 == len(topology) else int(topology[layerNum + 1])
				for neuronNum in range(int(topology[layerNum]) + 1):
					self.layers[-1].append(Neuron(numOutputs, neuronNum, layerNum, eta, alpha))
				self.layers[-1][-1].outputVal = 1
		else:
			self.layers = layers

	def feedForward(self, inputVals):
		assert len(inputVals) + 1 == len(self.layers[0])
		for i in range(len(inputVals)):
			self.layers[0][i].outputVal = inputVals[i]
		for layerNum in range(1, len(self.layers)):
			prevLayer = self.layers[layerNum - 1]
			for n in range(len(self.layers[layerNum]) - 1):
				self.layers[layerNum][n].feedForward(prevLayer)

	def backPropagate(self, targetVals):
		outputLayer = self.layers[-1]
		self.error = 0.0
		for n in range(len(outputLayer) - 1):
			self.error += (targetVals[n] - outputLayer[n].outputVal) * (targetVals[n] - outputLayer[n].outputVal)
		self.error = math.sqrt(self.error / (len(outputLayer) - 1))
		self.recentAverageError = (self.recentAverageError * self.recentAverageSmoothingFactor + self.error) / (self.recentAverageSmoothingFactor + 1.0);
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
		return {'__class__': 'Net', '__error__': self.error, '__recentAverageError__': self.recentAverageError, '__recentAverageSmoothingFactor__': self.recentAverageSmoothingFactor, '__layers__': [[neuron.toJSON() for neuron in layer] for layer in self.layers]}

	@classmethod
	def fromJSON(cls, JSON):
		if '__class__' in JSON:
			if JSON['__class__'] == 'Net':
				error = JSON['__error__']
				recentAverageError = JSON['__recentAverageError__']
				recentAverageSmoothingFactor = JSON['__recentAverageSmoothingFactor__']
				layers = [[Neuron.fromJSON(neuron) for neuron in layer] for layer in JSON['__layers__']]
				return Net(None, error, recentAverageError, recentAverageSmoothingFactor, layers)
