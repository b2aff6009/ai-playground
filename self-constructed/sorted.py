
import lib_nn as nn
import lib_draw as svg
import random

	
scale = 3.5
inputs = [
	{"values": [1, 2, 3, 5, 6, 7], "target": [1]},
	{"values": [2, 1, 3, 5, 6, 7, 8, 9], "target": [0]},
	{"values": [2, 2, 3, 5, 1, 5, 3, 2, 5], "target": [0]}
]

def costFunction(outputLayer : nn.Layer):
	return outputLayer.getValues()


def isTrivialSorted(array : list, target : list):

	sortedNet = nn.Net(costFunction)
	#create input layer, every node will just have the input value as value
	startLayer = nn.InputLayer(array)
	sortedNet.addLayer(startLayer)


	#create the first operation layer, which will check if the "second node" is higer than the "first node"
	nextLayer = nn.Layer(0.5, nn.relu, sortedNet.layerCnt(), 0.5)
	for i in range(0, len(array)-1):
		nodes = [sortedNet.prevLayer().nodes[i], sortedNet.prevLayer().nodes[i+1]]
		nextLayer.createNode(nodes, [-1, 1])
	sortedNet.addLayer(nextLayer)

	#second hidden layer will normalize the result of first hidden layer
	nextLayer = nn.Layer(-0.5, nn.sigmoid, sortedNet.layerCnt(), 0.5)
	for i in range(0, len(array)-1):
		nodes = [sortedNet.prevLayer().nodes[i]]
		nextLayer.createNode(nodes, [100])
	sortedNet.addLayer(nextLayer)

	#thirdLayer will just sum up all values from the second layer
	nextLayer = nn.Layer(0.0, nn.relu, sortedNet.layerCnt(), -0.5 + len(array)/2)
	nextLayer.createNode(sortedNet.prevLayer().nodes, [1]*len(array))
	sortedNet.addLayer(nextLayer)

	#output layer will normalize the sum from thirdLayer
	nextLayer= nn.Layer(-0.0, nn.relu, sortedNet.layerCnt(), -0.5 + len(array)/2)
	nextLayer.createNode(sortedNet.prevLayer().nodes, [1/(len(array)-1)])
	sortedNet.addLayer(nextLayer)

	print(sortedNet)
	print(sortedNet.getCost(target))
	return sortedNet

def isDenseSorted(array : list, target : list):

	sortedNet = nn.Net(costFunction)
	#create input layer, every node will just have the input value as value
	startLayer = nn.InputLayer(array)
	sortedNet.addLayer(startLayer)


	#create the first operation layer, which will check if the "second node" is higer than the "first node"
	nextLayer = nn.Layer(0.5, nn.relu, sortedNet.layerCnt(), 0.5)
	for i in range(0, len(array)-1):
		cnt = len(sortedNet.prevLayer().nodes)
		nextLayer.createNode(sortedNet.prevLayer().nodes, [(random.random()-0.5)*2]*cnt)
	sortedNet.addLayer(nextLayer)

	#second hidden layer will normalize the result of first hidden layer
	nextLayer = nn.Layer(-0.5, nn.sigmoid, sortedNet.layerCnt(), 0.5)
	cnt = len(sortedNet.prevLayer().nodes)
	for i in range(0, len(array)-1):
		nextLayer.createNode(sortedNet.prevLayer().nodes, [(random.random()-0.5)*2]*cnt)
	sortedNet.addLayer(nextLayer)

	#thirdLayer will just sum up all values from the second layer
	nextLayer = nn.Layer(0.0, nn.relu, sortedNet.layerCnt(), -0.5 + len(array)/2)
	cnt = len(sortedNet.prevLayer().nodes)
	nextLayer.createNode(sortedNet.prevLayer().nodes, [(random.random()-0.5)*2]*cnt)
	sortedNet.addLayer(nextLayer)

	#output layer will normalize the sum from thirdLayer
	nextLayer= nn.Layer(-0.0, nn.relu, sortedNet.layerCnt(), -0.5 + len(array)/2)
	cnt = len(sortedNet.prevLayer().nodes)
	nextLayer.createNode(sortedNet.prevLayer().nodes, [1/(len(array)-1)]*cnt)
	sortedNet.addLayer(nextLayer)

	# print(sortedNet)
	# print(sortedNet.getCost(target))
	return sortedNet


if __name__ == '__main__':
	for [i, input] in enumerate(inputs):
		net = isDenseSorted(input["values"], input["target"])
		print(net)
		# net = isTrivialSorted(input["values"], input["target"])
		svg.drawNet("sort_net_{}_1.svg".format(i), net, scale)

		correction = list()
		for layer in net.layers:
			if type(layer) is nn.InputLayer:
				continue
			layer_correction = list()
			for node in layer.nodes:
				node_correction = list()
				for weight in node.weights:
					node_correction.append(-weight)
				layer_correction.append(node_correction)
			correction.append(layer_correction)


		net.applyCorrection(correction)
		print(net)
		svg.drawNet("sort_net_{}_2.svg".format(i), net, scale)