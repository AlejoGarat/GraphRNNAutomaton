import numpy as np
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pythautomata.base_types.state import State
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.utilities.dfa_minimizer import DFAMinimizer
from automata_converter import automata_to_pythautomata_automata
from pythautomata.model_exporters.dot_exporters.dfa_dot_exporting_strategy import DfaDotExportingStrategy as DotExporter

def validate_property(property, automaton):
    return validate_automaton_property(property,
                            automata_to_pythautomata_automata(automaton))

def validate_automaton_property(property: str, automaton: DFA) -> bool:
    if property == "connected":
        return automaton_is_connected(automaton)
    elif property == "unique_accepting":
        return automaton_has_unique_accepting_state(automaton)
    elif property == "minimal":
        return automaton_is_minimal(automaton)
    else:
        raise ValueError("Unknown property")

def get_metrics(property, automatas):
    dfas = [automata_to_pythautomata_automata(automata) for automata in automatas]
    
    if property == "connected":
        return connected_automatas_metrics(dfas)
    elif property == "unique_accepting":
        return unique_accepting_automatas_metrics(dfas)
    elif property == "minimal":
        return minimal_automatas_metrics(dfas)
    else:
        raise ValueError("Unknown property")

def connected_automatas_metrics(automatas: [DFA]) -> dict:
    automatas_dead_end_states = []
    accuracy = 0
    for automata in automatas:
        is_connected, obs = automaton_is_connected(automata)
        accuracy += is_connected
        automatas_dead_end_states.append(obs["dead_end_states"])

    return {
        "mean_dead_end_states": np.mean(automatas_dead_end_states),
        "accuracy": accuracy / len(automatas) * 100,
    }
    
def unique_accepting_automatas_metrics(automatas: [DFA]) -> dict:
    final_states = []
    accuracy = 0
    for automata in automatas:
        is_unique, obs = automaton_has_unique_accepting_state(automata)
        accuracy += is_unique
        final_states.append(obs["final_states"])
        
    return {
        "mean_accepting_states": np.mean(final_states),
        "accuracy": accuracy / len(automatas) * 100,
    }
    
def minimal_automatas_metrics(automatas: [DFA]) -> dict:
    indistinguishable_states = []
    dead_end_states = []
    unreachable_states = []
    accuracy = 0 
    for automata in automatas:
        is_minimal, obs = automaton_is_minimal(automata)
        indistinguishable_states.append(obs["indistinguishable_states"])
        dead_end_states.append(obs["dead_end_states"])
        unreachable_states.append(obs["unreachable_states"])
        accuracy += is_minimal
            
    return {
        "mean_indistinguishable_states": np.mean(indistinguishable_states),
        "mean_dead_end_states": np.mean(dead_end_states),
        "mean_unreachable_states": np.mean(unreachable_states),
        "accuracy": accuracy / len(automatas) * 100
    }

def automaton_is_connected(automaton: DFA) -> tuple[bool, dict]:
    states = automaton.states
    is_connected = True
    amount = 0
    for state in states:
        if is_dead_end_state(automaton, state) and not(state.is_final):
            is_connected = False
            amount += 1
        
    obs = {
        "dead_end_states": amount
    }
    return is_connected, obs
        
def is_dead_end_state(dfa: DFA, state: State) -> bool:
    for symbol in dfa.alphabet.symbols:
        if  symbol not in state.transitions or state not in state.transitions[symbol]:
            return False
    
    return True

def automaton_has_unique_accepting_state(automaton: DFA) -> tuple[bool, dict]:
    final_states = 0
    for state in automaton.states:
        if state.is_final:
            final_states +=1
    obs = {
        "final_states": final_states
    }
    
    return final_states == 1, obs
                                                                              
    
def automaton_is_minimal(automaton: DFA) -> tuple[bool, dict]:
    dfa_minimizer = DFAMinimizer(automaton)
    final_eq_class, _ = dfa_minimizer._get_final_eq_class()
    indist_states_amount = get_dfa_indistinguishable_states_amount(final_eq_class)
    non_final_states = [state for state in automaton.states if not(state.is_final)]
    obs = {
        "equivalences_classes": len(final_eq_class),
        "indistinguishable_states": indist_states_amount,
        "dead_end_states": len([state for state in non_final_states if is_dead_end_state(automaton, state)]),
        "unreachable_states": get_unreachable_states_amount(automaton)
    }
    return len(automaton.states) == len(dfa_minimizer.minimize().states), obs

def get_dfa_indistinguishable_states_amount(final_eq_class: []) -> tuple[int, int]:
    amount = 0
    for eq_class in final_eq_class:
        if len(eq_class) > 1:
            amount += len(eq_class)
            
    return amount

def get_unreachable_states_amount(dfa: DFA) -> int:
    queue = []
    vis = []
    initial_state = dfa.initial_state
    queue.append(initial_state)
    
    while len(queue) > 0:
        node = queue.pop(0)
        vis.append(node)
        for neighbor in get_neighbors(dfa, node):
            if neighbor not in vis and neighbor not in queue:
                queue.append(neighbor)
    return len(dfa.states) - len(vis)
    
def get_neighbors(dfa: DFA, state: State) -> [State]:
    neighbors = []
    for symbol in dfa.alphabet.symbols:
        neigbor = state.next_state_for(SymbolStr(symbol))
        if neigbor not in neighbors:
            neighbors.append(neigbor)
            
    return neighbors
        
# Minimal property automata 
'''
state = State('q0')
alphabet = Alphabet([SymbolStr("a"), SymbolStr("b")])
state.add_transition(SymbolStr("a"), state)
state.is_final = False
dfa = DFA(alphabet, state, [state], None, None)

state = State('q0')
hole = State('q1', is_final=True)
alphabet = Alphabet([SymbolStr('0'), SymbolStr('1')])
state.is_final = False
dfa2 = DFA(alphabet, state, set([state]), None, None, hole=hole)
print("States: ", dfa2.states)
for state in dfa2.states:
    print("State: ", state)
    print("Transitons: ", state.transitions)

DotExporter().export(dfa2)
'''