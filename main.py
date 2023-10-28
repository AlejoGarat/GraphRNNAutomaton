import networkx as nx
import random
from matplotlib import pyplot as plt

G = nx.DiGraph()
num_nodes = 10
desired_out_degree = 2

nodes = list(range(num_nodes))

for node in nodes:
    possible_targets = nodes.copy()
    possible_targets.remove(node)

    # Limita las conexiones a una ventana de 3 nodos hacia adelante
    min_target = max(node + 1, node - 3)
    max_target = node + 3

    valid_targets = [target for target in possible_targets if min_target <= target <= max_target]
    
    if len(valid_targets) < desired_out_degree:
        targets = valid_targets
    else:
        targets = random.sample(valid_targets, desired_out_degree)

    G.add_edges_from([(node, target) for target in targets])

# Visualización del grafo
nx.draw(G, with_labels=True, node_size=300, node_color="skyblue", font_size=10)
plt.show()

adj_matrix = nx.to_numpy_array(G)

print("Matriz de Adyacencia:")
print(adj_matrix)