import svgwrite
from svgwrite import cm, mm   

import lib_nn as nn

outputFolder = "outputs"
def drawNet(name, net : nn.Net, scale : float):

	def drawLayer(dwg, layer : nn.Layer):
		svg_connections = dwg.add(dwg.g(id='connections', stroke='green'))
		svg_nodes = dwg.add(dwg.g(id='nodes', fill='red'))
		svg_values = dwg.add(dwg.g(id='value', fill='black'))

		if type(layer) is not nn.InputLayer: 
			weights = layer.getAllWeights()
			min_weight = min(weights)
			max_weight = max(weights)
		else:
			weights = 0
			min_weight = 0
			max_weight = 0

		def drawNodeConnection(node1 : nn.Node, node2 : nn.Node, weight: float, color : str):
			[x1 , y1] = node1.pos(scale)
			[x2 , y2] = node2.pos(scale)
			line = dwg.line(start=(x1*cm, y1*cm), end=((x2*cm, y2*cm)), stroke=color)
			svg_connections.add(line)

			# mx : float = (x2 + x1)/2
			# my : float = (y2 + y1)/2
			# text = dwg.text(f"{weight:.2f}", insert = (mx*cm, my*cm))
			# svg_connections.add(text)

		def drawNode(node : nn.Node, min_weight : float, max_weight : float):
			[x, y] = node.pos(scale)
			circle = dwg.circle(center=(x*cm, y*cm), r='0.6cm')
			svg_nodes.add(circle)
			if len(node.inputNodes) != 0:
				for [prevNode, weight] in zip(node.inputNodes, node.weights):
					color = "#000000"
					if weight < 0:
						rel_color = int(weight/min_weight * 255)
						color = '#{:0>2X}0000'.format(rel_color)
					else:
						rel_color = int(weight/max_weight * 255)
						color = '#00{:0>2X}00'.format(rel_color)

					drawNodeConnection(node, prevNode, weight, color)
			text = dwg.text(str(node), insert = (x*cm, y*cm))
			svg_values.add(text)
			
		for node in layer.nodes:
			drawNode(node, min_weight, max_weight)


	dwg = svgwrite.Drawing(filename=(outputFolder + "/" + name), debug=True)

	for layer in net.layers:
		drawLayer(dwg, layer)

	dwg.save()