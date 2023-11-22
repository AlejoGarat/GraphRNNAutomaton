import numpy as np
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pythautomata.base_types.state import State
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.utilities.dfa_minimizer import DFAMinimizer
from automata_converter import automata_to_pythautomata_automata

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
    
def connected_automatas_metrics(automatas: [DFA]) -> dict:
    automatas_dead_end_states = []
    automatas_with_dead_end_states = []
    for automata in automatas:
        is_connected, obs = automaton_is_connected(automata)
        if not is_connected:
            automatas_with_dead_end_states.append(1)
        else:
            automatas_with_dead_end_states.append(0)
        
        automatas_dead_end_states.append(obs["dead_end_states"])
    
    mae_dead_end_states_amount = np.mean(automatas_dead_end_states)
    mae_automatas_with_dead_end_states = np.mean(automatas_with_dead_end_states)
        
    return {
        "mae_dead_end_states_amount": mae_dead_end_states_amount,
        "mae_automatas_with_dead_end_states": mae_automatas_with_dead_end_states
    }
    
def unique_accepting_automatas_metrics(automatas: [DFA]) -> dict:
    final_states = []
    automatas_with_non_unique_accepting_state = []
    for automata in automatas:
        is_unique, obs = automaton_has_unique_accepting_state(automata)
        if is_unique:
            automatas_with_non_unique_accepting_state.append(1)
        else:
            automatas_with_non_unique_accepting_state.append(0)
        
        final_states.append(obs["final_states"])
        
    mae_final_states_amount = np.mean(final_states)
    mae_automatas_with_unique_accepting_state = np.mean(automatas_with_non_unique_accepting_state)
    
    return {
        "mae_unique_accepting_states_amount": mae_final_states_amount,
        "mae_automatas_with_unique_accepting_state": mae_automatas_with_unique_accepting_state
    }
    
def minimal_automatas_metrics(automatas: [DFA]) -> dict:
    indistinguishable_states = []
    dead_end_states = []
    unreachable_states = []
    for automata in automatas:
        _, obs = automaton_is_minimal(automata)
        indistinguishable_states.append(obs["indistinguishable_states"])
        dead_end_states.append(obs["dead_end_states"])
        unreachable_states.append(obs["unreachable_states"])
        
    mae_indistinguishable_states = np.mean(indistinguishable_states)
    mae_dead_end_states = np.mean(dead_end_states)
    mae_unreachable_states = np.mean(unreachable_states)
    
    return {
        "mae_indistinguishable_states": mae_indistinguishable_states,
        "mae_dead_end_states": mae_dead_end_states,
        "mae_unreachable_states": mae_unreachable_states
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
        if state not in state.transitions[symbol]:
            return False
    
    return True

def automaton_has_unique_accepting_state(automaton: DFA) -> tuple[bool, dict]:
    obs = {
        "final_states": automaton.final_states
    }
    return len(automaton.final_states) == 1, obs
                                                    
    
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
    for symbol in dfa.alphabet.symbols:
        yield state.next_state_for(symbol)
  
  
# Minimal property automata 
state = State("q0")
alphabet = Alphabet([SymbolStr("a"), SymbolStr("b")])
state.add_transition(SymbolStr("a"), state)
state.add_transition(SymbolStr("b"), state)
state.is_final = False
dfa = DFA(alphabet, state, [state], None, None)
state = State("q0")
alphabet = Alphabet([SymbolStr("a"), SymbolStr("b")])
state.add_transition(SymbolStr("a"), state)
state.add_transition(SymbolStr("b"), state)
state.is_final = False
dfa2 = DFA(alphabet, state, [state], None, None)

print(connected_automatas_metrics([dfa, dfa2]))
