
import lib_nn as nn
import lib_draw as svg
import random
	
scale = 3.5

min_start = 0
max_start = 10
def createInputs(lenght : int, count : int) -> list():
	inputs = list()
	for i in range(0, count):
		entry = dict()
		start = random.randint(min_start, max_start)
		values = list(range(start, start + lenght))
		target = 1
		if random.random() < 0.5:
			random.shuffle(values)
			target = 0
		entry["values"] = values
		entry["target"] = target
		inputs.append(entry)
	return inputs

inputs = createInputs(5, 1)

def costFunction(outputLayer : nn.Layer):
	return outputLayer.getValues()

def ManualNet(inputs : list[dict]):
	inputLen = len(inputs[0]["values"])
	net = nn.Net(inputLen, costFunction)
	net.setInput(inputs[0]["values"])

	net.addLayer(nn.NeightboursLayer(net, inputLen-1, 0.5, nn.relu, net.layerCnt(), 0.5, [-1, 1])) #get diff
	net.addLayer(nn.NeightboursLayer(net, inputLen-1, -0.5, nn.sigmoid, net.layerCnt(), 0.5, [10])) # normalize
	net.addLayer(nn.DenseLayer(net, 1, 0.0, nn.relu, net.layerCnt(), -0.5 + inputLen/2, [1]*(inputLen-1))) # sum up
	net.addLayer(nn.NeightboursLayer(net, 1, 0.0, nn.relu, net.layerCnt(), -0.5 + inputLen/2, [1/(inputLen-1)])) # normalize 

	return net

def RandomNet(inputs : list[dict]):
	inputLen = len(inputs[0]["values"])

	net = nn.Net(inputLen, costFunction)
	net.setInput(inputs[0]["values"])

	net.addLayer(nn.DenseLayer(net, inputLen-1, 0.5, nn.relu, net.layerCnt(), 0.5))
	net.addLayer(nn.DenseLayer(net, inputLen-1, -0.5, nn.sigmoid, net.layerCnt(), 0.5))
	net.addLayer(nn.DenseLayer(net, 1, 0.0, nn.relu, net.layerCnt(), -0.5 + inputLen/2))

	return net

def Execute( netType : int, inputs : list):
	if netType == 1:
		net = RandomNet(inputs)
		print(net)
		output = svg.Output(net, scale, "sort_net.svg")
		output.addAnimationStep()

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
		output.addAnimationStep()
		print(net)
		output.finish()

	elif netType == 0:
		net = ManualNet(inputs)
		print(net)
		output = svg.Output(net, scale, "sort_net.svg")
		output.addAnimationStep()
		output.finish()


if __name__ == '__main__':
	Execute(1, inputs)