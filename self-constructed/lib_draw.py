import svgwrite
from svgwrite import cm, mm   

import lib_nn as nn

outputFolder = "outputs"

lineWidth=2
def drawNet(name, net : nn.Net, scale : float):

	dwg = svgwrite.Drawing(filename=(outputFolder + "/" + name), debug=True)
	connections_group = dwg.g(id='connections', stroke='green')
	nodes_group = dwg.g(id='nodes', fill='red')
	values_group = dwg.g(id='value', fill='black')

	def drawLayer(dwg, layer : nn.Layer):

		if type(layer) is not nn.InputLayer: 
			weights = layer.getAllWeights()
			min_weight = min(weights)
			max_weight = max(weights)
			values= layer.getValues()
			max_value = max(values)
			max_value = max(1, max_value)
		else:
			weights = 0
			min_weight = 0
			max_weight = 0
			max_value = 0

		def drawNodeConnection(node1 : nn.Node, node2 : nn.Node, weight: float, color : str):
			[x1 , y1] = node1.pos(scale)
			[x2 , y2] = node2.pos(scale)
			line = dwg.line(start=(x1*cm, y1*cm), end=((x2*cm, y2*cm)), stroke=color, stroke_width=lineWidth)
			line.set_desc(title=f"{weight:.2f}")
			connections_group.add(line)

		def drawNode(node : nn.Node, max_value : float, min_weight : float, max_weight : float):
			[x, y] = node.pos(scale)
			min_color = 30
			if max_value == 0:
				rel_color = int(0.5 * (255-min_color))+min_color
			else:
				rel_color = int(node.get()/max_value * (255-min_color))+min_color

			circ_color= '#{:0>2X}{:0>2X}{:0>2X}'.format(rel_color, rel_color, rel_color)
			circle = dwg.circle(center=(x*cm, y*cm), r='0.6cm', fill=circ_color)
			circle.set_desc(title=str(node))

			nodes_group.add(circle)
			if len(node.inputNodes) != 0:
				for [prevNode, weight] in zip(node.inputNodes, node.weights):
					connection_color = "#000000"
					if weight < 0:
						rel_color = int(weight/min_weight * 255)
						connection_color = '#{:0>2X}0000'.format(rel_color)
					else:
						rel_color = int(weight/max_weight * 255)
						connection_color = '#00{:0>2X}00'.format(rel_color)

					drawNodeConnection(node, prevNode, weight, connection_color)
			value = dwg.text(str(node), insert = (x*cm, y*cm), text_anchor="middle")
			value.set_desc(title=str(node))
			values_group.add(value)
			
		for node in layer.nodes:
			drawNode(node, max_value, min_weight, max_weight)


	dwg.add(dwg.style("""
    circle:hover {
        stroke: black;
        stroke-width: 2;
		fill: lightgreen 
    }
"""))
	dwg.add(dwg.style("""
    line:hover {
        stroke: black;
        stroke-width: 4;
    }
"""))
	dwg.add(connections_group)
	dwg.add(nodes_group)
	dwg.add(values_group)

	for layer in net.layers:
		drawLayer(dwg, layer)

	dwg.save()