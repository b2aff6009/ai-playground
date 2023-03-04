
import lib_nn as nn
import lib_draw as svg

	
scale = 3.5
inputs = [
	[1, 2, 3, 5, 6, 7],
	[2, 1, 3, 5, 7, 8, 9],
	[2, 2, 3, 5, 1, 5, 4, 2, 5]
]

def isSorted(line):
	sortedNet = nn.Net()
	#create input layer, every node will just have the input value as value
	startLayer = nn.InputLayer(line)
	sortedNet.addLayer(startLayer)


	#create the first operation layer, which will check if the "second node" is higer than the "first node"
	nextLayer = nn.Layer(0.5, nn.relu, sortedNet.layerCnt(), 0.5)
	for i in range(0, len(line)-1):
		nodes = [sortedNet.prevLayer().nodes[i], sortedNet.prevLayer().nodes[i+1]]
		nextLayer.createNode(nodes, [-1, 1])
	sortedNet.addLayer(nextLayer)

	#second hidden layer will normalize the result of first hidden layer
	nextLayer = nn.Layer(-0.5, nn.sigmoid, sortedNet.layerCnt(), 0.5)
	for i in range(0, len(line)-1):
		nodes = [sortedNet.prevLayer().nodes[i]]
		nextLayer.createNode(nodes, [100])
	sortedNet.addLayer(nextLayer)

	#thirdLayer will just sum up all values from the second layer
	nextLayer = nn.Layer(0.0, nn.relu, sortedNet.layerCnt(), -0.5 + len(line)/2)
	nextLayer.createNode(sortedNet.prevLayer().nodes, [1]*len(line))
	sortedNet.addLayer(nextLayer)

	#output layer will normalize the sum from thirdLayer
	nextLayer= nn.Layer(-0.0, nn.relu, sortedNet.layerCnt(), -0.5 + len(line)/2)
	nextLayer.createNode(sortedNet.prevLayer().nodes, [1/(len(line)-1)])
	sortedNet.addLayer(nextLayer)

	print(sortedNet)
	return sortedNet


if __name__ == '__main__':
	for [i, input] in enumerate(inputs):
		svg.drawNet("sort_net_{}.svg".format(i), isSorted(input), scale)