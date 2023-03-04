import svgwrite
from svgwrite import cm, mm   

import lib_nn as nn

outputFolder = "outputs"
def drawNet(name, net : nn.Net, scale : float):
	def drawLayer(dwg, layer : nn.Layer):
		svg_connections = dwg.add(dwg.g(id='connections', stroke='green'))
		svg_nodes = dwg.add(dwg.g(id='nodes', fill='red'))
		svg_values = dwg.add(dwg.g(id='value', fill='black'))

		def drawNodeConnection(node1 : nn.Node, node2 : nn.Node):
			[x1 , y1] = node1.pos(scale)
			[x2 , y2] = node2.pos(scale)
			line = dwg.line(start=(x1*cm, y1*cm), end=((x2*cm, y2*cm)))
			svg_connections.add(line)

		def drawNode(node : nn.Node):
			[x, y] = node.pos(scale)
			circle = dwg.circle(center=(x*cm, y*cm), r='0.6cm')
			svg_nodes.add(circle)
			for prevNode in node.inputNodes:
				drawNodeConnection(node, prevNode)
			text = dwg.text(str(node), insert = (x*cm, y*cm))
			svg_values.add(text)
			
		for node in layer.nodes:
			drawNode(node)


	dwg = svgwrite.Drawing(filename=(outputFolder + "/" + name), debug=True)

	for layer in net.layers:
		drawLayer(dwg, layer)

	dwg.save()