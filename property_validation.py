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

def validate_automaton_property(property: str, automaton: DFA) -> bool:
    if property == "connected":
        return automaton_is_connected(automaton)
    elif property == "unique_accepting":
        return automaton_has_unique_accepting_state(automaton)
    elif property == "minimal":
        return automaton_is_minimal(automaton)
    else:
        raise ValueError("Unknown property")

def automaton_is_connected(automaton: DFA) -> bool:
    states = automaton.states
    for state in states:
        if is_hole_state(automaton, state):
            return False
        
    return True
        
def is_hole_state(dfa: DFA, state: State) -> bool:
    for symbol in dfa.alphabet.symbols:
        if state.transitions[symbol] != state:
            return False
    
    return True

def automaton_has_unique_accepting_state(automaton: DFA) -> bool:
    return len(automaton.final_states) == 1
    
    
def automaton_is_minimal(automaton: DFA) -> bool:
    pass