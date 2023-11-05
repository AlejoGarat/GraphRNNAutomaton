from automata import Automata
import networkx as nx

transitions = ...  # Definir las transiciones de tu autómata
final_states = ...  # Definir los estados finales de tu autómata
automaton = Automata(transitions, final_states)

# Crea un grafo dirigido a partir de las transiciones del autómata
automaton_graph = nx.DiGraph()

# Agrega los nodos y las transiciones al grafo
for state in range(len(transitions)):
    in_transitions, loop_transition, out_transitions = automaton.get_transitions(state)
    for target_state in in_transitions:
        automaton_graph.add_edge(target_state, state)
    if loop_transition:
        automaton_graph.add_edge(state, state)
    for target_state in out_transitions:
        automaton_graph.add_edge(state, target_state)

# Verifica la conectividad del grafo (autómata)
if nx.is_strongly_connected(automaton_graph):
    print("El autómata es conexo.")
else:
    print("El autómata no es conexo.")
