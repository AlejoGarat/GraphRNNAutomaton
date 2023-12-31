import numpy as np
from automata import Automata
from pythautomata.base_types.state import State
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pythautomata.model_exporters.dot_exporters.dfa_dot_exporting_strategy import DfaDotExportingStrategy as DotExporter
from pythautomata.model_exporters.image_exporters.image_exporting_strategy import ImageExportingStrategy

def convert_to_automata(g, alphabet):
    nodes = len(g.nodes.values())
    transitions = [[set() for _ in range(nodes)] for _ in range(nodes)]    
    for origin, dests in g.nodes.items():
        if len(dests) == 0:
            continue

        sampling_dests = dests.copy()
        was_emptied = False
        for symbol in alphabet:
            dest = np.random.choice(list(sampling_dests))
            transitions[origin][dest].add(symbol)
            if not was_emptied:
                sampling_dests.remove(dest)
                
            if len(sampling_dests) == 0:
                sampling_dests = dests.copy()
                was_emptied = True

    return Automata(transitions, {str(n) for n in g.final_nodes}, alphabet, 
                '0', {str(i):i for i in range(nodes)})

def automata_to_pythautomata_automata(automata: Automata) -> DFA:
    alphabet = automata.alphabet
    final_states = automata.final_states
    initial_state = State(automata.initial_state)
    states = set()
    states.add(initial_state)
    for state_name in automata.pos_dict.keys():
        if state_name != automata.initial_state:
            states.add(State(state_name, state_name in final_states))

    for state in states:
        state_pos = automata.pos_dict[state.name]
        for pos in range(len(automata.transitions)):
            for symbol in automata.transitions[state_pos][pos]:
                state.add_transition(SymbolStr(symbol), get_state_of_pos(pos, automata, states))      
    
    hole = State(name="hole", is_final=True)
    return DFA(alphabet=Alphabet([SymbolStr(str(s)) for s in list(alphabet)]), 
               states=states, 
               initial_state=initial_state, 
               comparator=None,
               hole=hole)
   
def get_state_of_pos(pos: int, automata: Automata, states: set):
    for state in states:
        if automata.pos_dict[state.name] == pos:
            return state
    return None
    
    