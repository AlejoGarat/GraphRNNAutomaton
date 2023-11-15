import numpy as np
from automata import Automata
from exporter import export_automatas
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pythautomata.base_types.state import State
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.utilities.nicaud_dfa_generator import generate_dfa
from pymodelextractor.factories.lstar_factory import LStarFactory
from pymodelextractor.teachers.general_teacher import GeneralTeacher
from pythautomata.model_comparators.hopcroft_karp_comparison_strategy import HopcroftKarpComparisonStrategy as Hopcroft
from pythautomata.model_comparators.dfa_comparison_strategy import DFAComparisonStrategy
from pythautomata.model_exporters.dot_exporters.dfa_dot_exporting_strategy import DfaDotExportingStrategy as DotExporter
from pythautomata.utilities.dfa_minimizer import DFAMinimizer

def validate_automaton_property(property: str, automaton: DFA) -> bool:
    if property == "connected":
        return automaton_is_connected(automaton)
    elif property == "unique_accepting":
        return automaton_has_unique_accepting_state(automaton)
    elif property == "minimal":
        return automaton_is_minimal(automaton)
    else:
        raise ValueError("Unknown property")

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
        if state.transitions[symbol] != state:
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
    obs = {
        "equivalences_classes": len(final_eq_class),
        "indistinguishable_states": indist_states_amount,
        "dead_end_states": len([state for state in automaton.states if is_dead_end_state(automaton, state)]),
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
    
automaton_is_minimal(None)