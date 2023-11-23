from automata_converter import automata_to_pythautomata_automata
from automata import Automata
import numpy as np
from pythautomata.model_exporters.dot_exporters.dfa_dot_exporting_strategy import DfaDotExportingStrategy as DotExporter
# Convert an automata to a pythautomata automata example

transitions = np.empty((3, 3), dtype=object)
transitions[0, 0] = set('a')
transitions[0, 1] = set('b')
transitions[0, 2] = set()
transitions[1, 0] = set()
transitions[1, 1] = set(['a', 'b'])
transitions[1, 2] = set()
transitions[2, 0] = set()
transitions[2, 1] = set('a')
transitions[2, 2] = set('b')
initial_state = '0'
pos_dict = {'0': 0, '1': 1, '2': 2}
final_states = {'2'}
alphabet = ['a', 'b']

automata = Automata(transitions, final_states, alphabet, initial_state, pos_dict)

pythautomata_automata = automata_to_pythautomata_automata(automata)
DotExporter().export(pythautomata_automata)








