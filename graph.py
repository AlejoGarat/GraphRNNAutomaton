import numpy as np

class Graph:
    def __init__(self):
        self.nodes = {}
        self.final_nodes = set()
        
    def add_node(self, node, conns, is_final):
        self.nodes[node] = set()
        m = (len(conns)-1)//2
        in_conns = conns[max(0, m-node):m]
        loop_p = float(conns[m])
        out_conns = conns[m+1:len(conns)-max(0,m-node)]

        for target, p_in in enumerate(in_conns):
            p_in = float(p_in)
            in_connection = np.random.choice([False, True], p=[1-p_in, p_in])
            if in_connection:
                self.nodes[target].add(node)

            p_out = float(out_conns[target])
            out_connection = np.random.choice([False, True], p=[1-p_out, p_out])
            if out_connection:
                self.nodes[node].add(target)
        
        loop_connection = np.random.choice([False, True], p=[1-loop_p, loop_p])
        if loop_connection:
            self.nodes[node].add(node)

        if is_final:
            self.final_nodes.add(node)