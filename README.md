#LEARNING

**Exploring Machine Learning: Neural Networks and Genetic Algorithms**

The code in this repository contains implementations of Distributed Neural
Networks exploring different degrees of training parallelism. Version 1
implements batch Gradient Descent over the divided dataset. Version 2 implements
stochastic Gradient Descent on the divided dataset, each set distributed to a
different network, followed by a combination using Genetic Algorithms to evolve
the best trained network. Both were written keeping the Map Reduce Paradigm in
mind for easy extension to distributed filesystems such as Apache's Hadoop.
Data is read from stdin and written to stdout to better utilize Hadoop's
Streaming API. The Hadoop implementations have not been uploaded.

###v1

**Shell Scripts**:

1. **driver.sh**: runs training and testing scripts

**Instructions**:

1. **Setting Executable Bit**: $chmod +x driver.sh
2. **Executing**: $./driver.sh [datafile] [fractiontest] [chunksize] [modelfile]
[eta] [alpha] [topology]

**Arguments**:

* **datafile**: (string) name of file from which data is to be read
* **fractiontest**: (float) fraction of data to be used for testing
* **chunksize**: (int) number of training samples in each batch
* **modelfile**: (string) name of file from/to which model is to be read/written
* **eta**: (float) neural network learning parameter
* **alpha**: (float) neural network learning parameter
* **topology**: (list: int) list of integers defining the number of neurons in
each layer

**Input File Format**:

Each line is a training or testing instance containing both inputs and outputs.
Each line contains a whitespace separated sequence of numbers. The 1st and 2nd
numbers are integers, denoting the number of inputs and outputs on the line,
respectively. They are followed by the specified number of floating point inputs
and outputs, in order.

**Note**:

* Requires BASH (or equivalent shell).
* Requires Python.

**Python Scripts**:

1. **dividedata.py**: divides data into training and test sets
2. **testmap.py**: mapper for testing distributed neural network
3. **testreduce.py**: reducer for testing distributed neural network
4. **trainmap.py**: mapper for training distributed neural network
5. **trainreduce.py**: reducer for training distributed neural network

**Note**:

* Requires Python

**Python Classes**:

1. **DNN.py**: distributed neural network implementation

**Note**:

* Requires Python

###v2

**Python Scripts**:

1. **learn.py**: runs the training and testing

**Instructions**:

1. **Setting Executable Bit**: $chmod +x learn.py
2. **Executing**: $./learn.py [datafile] [fractiontest] [parallel] [modelfile]
[crossover] [mutation] [generations] [extinction] [eta] [alpha] [topology]

**Arguments**:

* **datafile**: (string) name of file from which data is to be read
* **fractiontest**: (float) fraction of data to be used for testing
* **parallel**: (int) number of training samples for each network
* **modelfile**: (string) name of file from/to which model is to be read/written
* **crossover**: (float) genetic algorithm evolving parameter
* **mutation**: (float) genetic algorithm evolving parameter
* **generations**: (int) genetic algorithm evolving parameter
* **extinction**: (float) genetic algorithm evolving parameter
* **eta**: (float) neural network learning parameter
* **alpha**: (float) neural network learning parameter
* **topology**: (list: int) list of integers defining the number of neurons in
each layer

**Input File Format**:

Each line is a training or testing instance containing both inputs and outputs.
Each line contains a whitespace separated sequence of numbers. The 1st and 2nd
numbers are integers, denoting the number of inputs and outputs on the line,
respectively. They are followed by the specified number of floating point inputs
and outputs, in order.

**Note**:

* Requires Python.

**Python Classes**:

1. **NN.py**: neural network implementation
2. **GA.py**: genetic algorithm implementation

**Note**:

* Requires Python.
