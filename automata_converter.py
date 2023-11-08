from automata import Automata
from pythautomata.base_types.state import State 
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pythautomata.model_exporters.dot_exporters.dfa_dot_exporting_strategy import DfaDotExportingStrategy as DotExporter
from pythautomata.model_exporters.image_exporters.image_exporting_strategy import ImageExportingStrategy

def automata_to_pythautomata_automata(automata: Automata) -> DFA:
    alphabet = automata.alphabet
    final_states = automata.final_states
    initial_state = State(automata.initial_state)
    states = set()
    for state_name in automata.pos_dict.keys():
        states.add(State(state_name, state_name in final_states))

    for state in states:
        state_pos = automata.pos_dict[state.name]
        for pos in range(len(automata.transitions)):
            for symbol in automata.transitions[state_pos, pos]:
                state.add_transition(symbol, get_state_of_pos(pos, automata, states))      
                
    return DFA(alphabet=alphabet, states=states, initial_state=initial_state, comparator=None)
   
def get_state_of_pos(pos: int, automata: Automata, states: set):
    for state in states:
        if automata.pos_dict[state.name] == pos:
            return state
    return None
    
    