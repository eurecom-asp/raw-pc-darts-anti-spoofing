import sys
import genotypes
# import genotypes <- that was present in the original DARTS implementation. Not needed here
from graphviz import Digraph


def plot(genotype, filename):
  g = Digraph(
      format='png',
      edge_attr=dict(fontsize='40', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='40', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=TB'])
  g.attr(width="900", height="1800", fixedsize='true')

  input_node1 = 'X1'
  input_node2 = 'X2'
  g.node(input_node1, fillcolor='darkseagreen2')
  g.node(input_node2, fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  node_label = lambda i: f'X{i+3}'

  for i in range(steps):
    g.node(node_label(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = input_node1
      elif j == 1:
        u = input_node2
      else:
        u = node_label(j-2)
      v = node_label(i)
      g.edge(u, v, label=op, fillcolor="gray")

  output_node = f'X{steps+3}'
  g.node(output_node, fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(node_label(i), output_node, fillcolor="gray")

  g.render(filename, view=True)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  genotype_name = sys.argv[1]
  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name))
    sys.exit(1)

  plot(genotype.normal, "normal")
  plot(genotype.reduce, "reduction")
