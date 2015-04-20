import random, copy, sys

class NeuralGene:

	net = None
	gene = None
	testdata = None

	def __init__(self, net, testdata):
		self.net = net
		self.testdata = testdata
		self.gene = list()
		for i in range(len(self.net.layers)):
			for j in range(len(self.net.layers[i])):
				for k in range(len(self.net.layers[i][j].outputWeights)):
					self.gene.append(self.net.layers[i][j].outputWeights[k])

	def fitness(self):
		self.error = 0.0
		count = 0
		for line in open(self.testdata, mode = 'r').readlines():
			tokens = line.strip().split()
			numInputs = int(tokens[0])
			numOutputs = int(tokens[1])
			if len(tokens) == 2 + numInputs + numOutputs and numInputs + 1 == len(self.net.layers[0]) and numOutputs + 1 == len(self.net.layers[-1]):
				inputs = [float(token) for token in tokens[2 : 2 + numInputs]]
				outputs = [float(token) for token in tokens[2 + numInputs: 2 + numInputs + numOutputs]]
				self.net.feedForward(inputs)
				for index in range(len(self.net.layers[-1]) - 1):
					self.error += (outputs[index] - self.net.layers[-1][index].output) * (outputs[index] - self.net.layers[-1][index].output)
			count += 1
		self.averageError = self.error / float(count)
		if self.averageError == 0:
			return sys.float_info.max
		else:
			return 1.0 / self.averageError

	def spawn(self):
		i = 0
		for j in range(len(self.net.layers)):
			for k in range(len(self.net.layers[j])):
				for l in range(len(self.net.layers[j][k].outputWeights)):
					self.net.layers[j][k].outputWeights[l] = self.gene[i]
					i += 1

class Generation:

	crossover_rate = None
	mutation_rate = None
	extinction_rate = None
	population = None
	fitness = None

	def __init__(self, crossover_rate, mutation_rate, extinction_rate, population):
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		self.extinction_rate = extinction_rate
		self.population = population

	def evolve(self, size = None):
		if size == None:
			size = len(self.population)
		fitness = [i.fitness() for i in self.population]
		fitness, self.population = [list(x) for x in zip(*sorted(zip(fitness, self.population), key=lambda pair: pair[0], reverse=True))]
		survivalsize = len(self.population) - int(self.extinction_rate*len(self.population))
		self.population = self.population[0: survivalsize]
		fitness = fitness[0: survivalsize]
		new_population = list()
		for i in range(1, len(self.population)):
			fitness[i] += fitness[i-1]
		for i in range(len(self.population)):
			fitness[i] /= fitness[-1]
		for i in range(size):
			rand = random.random()
			j = 0
			k = 0
			for j in range(len(self.population)-1):
				if(fitness[j] > rand):
					break
			rand = random.random()
			for k in range(len(self.population)-1):
				if(fitness[k] > rand):
					break
			new_population.append(copy.deepcopy(self.population[0]))
			m = len(self.population[j].gene) if len(self.population[j].gene) < len(self.population[k].gene) else len(self.population[k].gene)
			c = True
			for l in range(m):
 				rand = random.random()
				c = not c if random < self.crossover_rate else c
				new_population[-1].gene[l] = self.population[j].gene[l] if c else self.population[k].gene[l]
			for l in range(m):
				rand = random.random()
				if rand < self.mutation_rate:
					new_population[-1].gene[l] == self.population[int(random.random() * len(self.population))].gene[l]
			new_population[-1].spawn()
		self.population = new_population
