import math
import random

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def relu(x):
  return 0 if x < 0 else x


class Net:
	def __init__(self, inputLen : int, costFunc):
		self.layers = []
		self.inputLen = inputLen
		self.costFunc = costFunc

		self.inputLayer = InputLayer(inputLen)
		self.addLayer(self.inputLayer)

	def addLayer(self, newLayer):
		self.layers.append(newLayer)
		return self.lastLayer()

	def lastLayer(self):
		return self.layers[-1]
	
	def layerCnt(self):
		return len(self.layers)	

	def __repr__(self):
		return str(self.layers)
	
	def applyCorrection(self, corrections: list):
		for [layer, correction] in zip(self.layers[1:], corrections):
			layer.applyCorrection(correction)
	
	def getCost(self, targetValues):
		result = []
		costs = self.costFunc(self.layers[-1])
		for i, j in zip(targetValues, costs):
			result.append((i -j)**2)
		return result
	
	def setInput(self, values : list[float]):
		self.inputLayer.setValues(values)

class Layer:
	def __init__(self, bias : float, activationFunc, nr : float = 0.0 , y_offset : float = 0):
		self.x : float = nr + 1
		self.y_offset : float = y_offset + 1
		self.bias : float = bias
		self.activationFunc = activationFunc
		self.nodes : list[Node] = []

	def setOffset(self, x : float, y_offset : float):
		self.y_offset = y_offset + 1
		self.x = x + 1

	def createNode(self, nodes, weights):
		self.nodes.append(Node(nodes, weights, self.bias, self.activationFunc, self.x, len(self.nodes) + self.y_offset))

	def applyCorrection(self, corrections: list):
		for [node, correction] in zip(self.nodes, corrections):
			node.applyCorrection(correction)

	def getValues(self):
		values = [node.get() for node in self.nodes]
		return values

	def __repr__(self):
		return str(self.nodes)
	
	def getAllWeights(self):
		weights = list()
		for node in self.nodes:
			weights.extend(node.weights)
		return weights
	
class InputLayer(Layer):
	def __init__(self, numberOfInputs: int):
		super().__init__(0.0, lambda x: 0, 0, 0.0)
		self.nodes = []
		for value in range(0, numberOfInputs):
			self.createNode(0.0)

	# def __init__(self, values : float):
	# 	super().__init__(0.0, lambda x: 0, 0, 0.0)
	# 	self.nodes = []
	# 	for value in values:
	# 		self.createNode(value)

	def createNode(self, value):
		self.nodes.append(InputNode(value, self.x, self.y_offset + len(self.nodes)))

	def getAllWeights(self):
		return []
	
	def setValues(self, values : list[float]):
		if len(values) != len(self.nodes):
			raise
		for [node, value] in zip(self.nodes, values):
			node.value = value


class AutoLayer(Layer):
	def __init__(self, net: Net, nodesCnt : int, bias : float, activationFunc, nr : float = 0.0 , y_offset : float = 0, weights : list = []):
		super().__init__(bias, activationFunc, nr, y_offset)
		self.net = net

		weightsCount : int = len(weights)
		for i in range(0, nodesCnt):
			nodes = self.getConnectedNodes(i, nodesCnt)
			if weightsCount == 0:
				for j in range(0, len(nodes)):
					weights.append((random.random() - 0.5)*2)
			self.createNode(nodes, weights)

class DenseLayer(AutoLayer):
	def getConnectedNodes(self, i : int, nodesCnt : int):
		return self.net.lastLayer().nodes

class NeightboursLayer(AutoLayer):
	def getConnectedNodes(self, position : int, nodesCnt : int):
		lastNodes : list[Node] = self.net.lastLayer().nodes
		neightbourFactor : float = len(lastNodes) / nodesCnt
		start : int = position * math.floor(neightbourFactor)
		diff : int = math.ceil(neightbourFactor)

		connectedNodes : list[Node] = list()
		for i in range(start, start + diff):
			connectedNodes.append(lastNodes[i])
		
		return connectedNodes


class Node:
	def __init__(self, nodes, weigths, bias, activation, x, y):
		self.x = x
		self.y = y
		self.inputNodes = nodes
		self.weights = weigths
		self.value = 0
		self.executed = False
		self.bias = bias
		self.activation = activation
	
	def __repr__(self):
		return f"{self.get():.2f}"

	def applyCorrection(self, corrections: list):
		self.executed = False
		for [i, correction] in enumerate(corrections):
			self.weights[i] = correction

	def get(self):
		if self.executed:
			return self.value

		nodesValue = 0
		for [i, node] in enumerate(self.inputNodes):
			nodesValue += node.get() * self.weights[i]
		self.value = self.activation(nodesValue + self.bias)
		self.executed = True

		return self.value
	
	def pos(self, scale):
		return [self.x * scale, self.y * scale]
	

class InputNode(Node):
	def __init__(self, value, x, y):
		self.x = x
		self.y = y
		self.value = value
		self.inputNodes = []
		self.executed = True 

	def get(self):
		return self.value