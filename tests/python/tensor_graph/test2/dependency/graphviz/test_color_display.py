from graphviz import Graph


g = Graph(format="svg")

with open("color_svg.txt", "r") as fin:
  name = 1
  for color in fin:
    color = color.strip()
    g.node(str(name) + "_" + color, color=color, fontcolor="white", style="filled")
    name += 1

g.view()