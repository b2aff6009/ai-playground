
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def relu(x):
  return 0 if x < 0 else x

class Net:
	def __init__(self):
		self.layers = []
		pass

	def addLayer(self, newLayer):
		self.layers.append(newLayer)

	def prevLayer(self):
		return self.layers[-1]
	
	def layerCnt(self):
		return len(self.layers)

	def __repr__(self):
		return str(self.layers)

class Layer:
	def __init__(self, bias, activationFunc, nr = 0.0 , y_offset = 0):
		self.x = nr + 1
		self.y_offset = y_offset + 1
		self.bias = bias
		self.activationFunc = activationFunc
		self.nodes = []

	def createNode(self, nodes, weights):
		self.nodes.append(Node(nodes, weights, self.bias, self.activationFunc, self.x, len(self.nodes) + self.y_offset))

	def __repr__(self):
		return str(self.nodes)
	
class InputLayer(Layer):
	def __init__(self, values):
		super().__init__(0.0, lambda x: 0, 0, 0.0)
		self.nodes = []
		for value in values:
			self.createNode(value)

	def createNode(self, value):
		self.nodes.append(InputNode(value, self.x, self.y_offset + len(self.nodes)))


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

	def get(self):
		return self.value