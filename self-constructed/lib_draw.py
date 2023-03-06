import svgwrite
from svgwrite import cm, mm   

import lib_nn as nn

outputFolder = "outputs"

lineWidth=2

class Output:
	def __init__(self, net : nn.Net, scale : float, name):
		self.net = net
		self.scale = scale 
		self.dwg = svgwrite.Drawing(filename=(outputFolder + "/" + name), debug=True)

		self.dwg.add(self.dwg.style("""
		circle:hover {
			stroke: black;
			stroke-width: 2;
			fill: lightgreen 
		}
	"""))
		self.dwg.add(self.dwg.style("""
		line:hover {
			stroke: black;
			stroke-width: 4;
		}
	"""))
		# self.connections_group = self.dwg.g(stroke='green')
		# self.nodes_group = self.dwg.g(id='nodes', fill='red')
		# self.values_group = self.dwg.g(id='value', fill='black')
		self.animations = list()

	#created by ChatGPT
	def invert_color(self, hex_color):
		# remove the '#' from the color string
		hex_color = hex_color.lstrip('#')
		# convert the hex color string to an integer
		hex_value = int(hex_color, 16)
		# invert the color by toggling the bits using bitwise operations
		inverted_hex = hex_value ^ 0xFFFFFF
		# convert the inverted hex color value back to a string
		inverted_color = '#' + format(inverted_hex, '06x')
		return inverted_color

	def drawNodeConnection(self, node1 : nn.Node, node2 : nn.Node, weight: float, color : str, connections_group):
		[x1 , y1] = node1.pos(self.scale)
		[x2 , y2] = node2.pos(self.scale)
		line = self.dwg.line(start=(x1*cm, y1*cm), end=((x2*cm, y2*cm)), stroke=color, stroke_width=lineWidth)
		line.set_desc(title=f"{weight:.2f}")
		connections_group.add(line)

	def drawNode(self, node : nn.Node, max_value : float, min_weight : float, max_weight : float, connections_group, nodes_group, values_group):
		[x, y] = node.pos(self.scale)
		min_color = 30
		if max_value == 0:
			rel_circ_color = int(0.5 * (255-min_color))+min_color
		else:
			rel_circ_color = int(node.get()/max_value * (255-min_color))+min_color

		circ_color= '#{:0>2X}{:0>2X}{:0>2X}'.format(rel_circ_color, rel_circ_color, rel_circ_color)
		circle = self.dwg.circle(center=(x*cm, y*cm), r='0.6cm', fill=circ_color)
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

				self.drawNodeConnection(node, prevNode, weight, connection_color, connections_group)

		text_color = self.invert_color(circ_color)
		value = self.dwg.text(str(node), insert = (x*cm, y*cm), text_anchor="middle", fill=text_color)
		value.set_desc(title=str(node))
		values_group.add(value)

	def drawLayer(self, dwg, layer : nn.Layer, connections_group, nodes_group, values_group):

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
	
		for node in layer.nodes:
			self.drawNode(node, max_value, min_weight, max_weight, connections_group, nodes_group, values_group)


	def drawNet(self, net : nn.Net, connections_group, nodes_group, values_group):
		for layer in net.layers:
			self.drawLayer(self.dwg, layer, connections_group, nodes_group, values_group)

	def addAnimationStep(self):

		animation_group = self.dwg.add(self.dwg.g(id='outer_group', opacity=0))

		connections_group = animation_group.add(self.dwg.g(stroke='green'))
		nodes_group = animation_group.add(self.dwg.g(id='nodes', fill='red'))
		values_group = animation_group.add(self.dwg.g(id='value', fill='black'))

		connections_group["class"] = "connections"
		nodes_group["class"] = "nodes"
		values_group["class"] = "values"

		self.drawNet(self.net, connections_group, nodes_group, values_group)

		self.animations.append(animation_group)

	def finish(self):
		for [b, animation] in enumerate(self.animations):
			startTime = b*2
			animation.add(self.dwg.animate('opacity', from_=0, to=1, dur='1s', begin=f'{startTime}s' ))
			animation.add(self.dwg.animate('opacity', from_=1, to=1, dur='1s', begin=f'{startTime+1}s', fill='freeze'))
			if b+1 != len(self.animations):
				animation.add(self.dwg.animate('opacity', from_=1, to=0, dur='1s', begin=f'{startTime+2}s' ))

			self.dwg.add(animation)
		self.dwg.save()