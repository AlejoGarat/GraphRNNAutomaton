import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import RNN, Linear, Dropout
from graph import Graph
from automata_converter import convert_to_automata

class EdgeMLP(nn.Module):
    def __init__(self, m, input_dim):
        super(EdgeMLP, self).__init__()
        self.m = m

        self.l1 = Linear(in_features=input_dim, out_features=512)
        self.l2 = Linear(in_features=512, out_features=256)
        self.l3 = Linear(in_features=256, out_features=512)
        self.l4 = Linear(in_features=512, out_features=2*m+3)
    
        self.dropout = Dropout(p=.3)

    def forward(self, x):
        res = F.sigmoid(self.l1(x))
        res = self.dropout(res)
        res = F.leaky_relu(self.l2(res), negative_slope=.02)
        res = self.dropout(res)
        res = F.leaky_relu(self.l3(res), negative_slope=.02)
        res = self.dropout(res)
        res = F.sigmoid(self.l4(res))

        return res
    
class NodeRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0):
        super(NodeRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim 
        self.rnn = RNN(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x, h):
        _, h = self.rnn(x, h)
        return h[-1]
    
    def get_sos(self, batch_size):
        return torch.zeros((batch_size, 1, self.input_dim))
    
    def get_initial_hidden(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_dim))
    
class AutomatonRNN(nn.Module):
    def __init__(self, m, hidden_dim):
        super(AutomatonRNN, self).__init__()
        self.m = m
        self.node_rnn = NodeRNN(2*m+3, hidden_dim)
        self.edge_model = EdgeMLP(m, input_dim=hidden_dim)

    def forward(self, x, h):
        hidden = self.node_rnn(x, h)
        return self.edge_model(hidden), hidden
    
    def get_sos(self, n):
        return self.node_rnn.get_sos(n)

    def get_initial_hidden(self, n):
        return self.node_rnn.get_initial_hidden(n)
    
def generate(model, max_nodes, number_of_graphs):
    with torch.no_grad():
        graphs = [Graph() for _ in range(number_of_graphs)]
        ends = [False for _ in range(number_of_graphs)]
        nodes = [0 for _ in range(number_of_graphs)]
        end = False
        x = model.get_sos(number_of_graphs)
        h = model.get_initial_hidden(number_of_graphs)

        while not end:
            x, h = model(x, h)
            conns, final_probs, end_probs = unfold_pred(x, model.m)
            for i, final_prob in enumerate(final_probs):
                if not ends[i]:
                    final_prob = float(final_prob)
                    end_prob =  float(end_probs[i])
                    is_final = np.random.choice([False, True], p=[1-final_prob, final_prob])
                    graphs[i].add_node(nodes[i], conns[i], is_final)
                    ends[i] = np.random.choice([False, True], p=[1-end_prob, end_prob])
                    nodes[i] += 1

            end = np.sum(ends) == len(ends)
            if nodes[0] > max_nodes:
                end = True
                
            x = x.reshape(number_of_graphs, 1, -1)
            h = h.reshape(1, number_of_graphs, -1)

        return graphs

def generate_automatas(model, max_nodes, number_graphs, alphabet_len):
    return [convert_to_automata(g, [str(i) for i in range(alphabet_len)]) for g in generate(model, max_nodes, number_graphs)]
    
def unfold_pred(res, m):
    conns = res[:,:2*m+1]
    final_prob = res[:,2*m+1]
    end_prob = res[:,2*m+2]
    return conns, final_prob, end_prob