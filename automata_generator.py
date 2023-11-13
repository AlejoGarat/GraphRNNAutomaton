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

def generate_and_export_automatas(property: str, amount: int, nominal_size: int, alphabet_size: int, path: str):
    dfas = generate_many_automatas(amount, nominal_size, alphabet_size)
    automatas = []
    if property == "connected":
        for dfa in dfas:
            automatas.append(pythautomata_automata_to_automata(make_automata_connected(dfa)))
    elif property == "unique_accepting":
        for dfa in dfas:
            automatas.append(pythautomata_automata_to_automata(make_automata_unique_accepting(dfa)))
    elif property == "minimal":
        for dfa in dfas:
            automatas.append(pythautomata_automata_to_automata(make_automata_minimal(dfa)))
    else:
        raise ValueError("Unknown property")

    export_automatas(automatas, path)

def generate_many_automatas(amount: int, nominal_size: int, alphabet_size: int) -> [DFA]:
    alphabet = Alphabet([SymbolStr(str(i)) for i in range(alphabet_size)])
    generated_automatas = []
    for _ in range(amount):
        generated_automatas.append(generate_dfa(alphabet=alphabet, nominal_size=nominal_size))
        
    return generated_automatas

def pythautomata_automata_to_automata(dfa: DFA) -> Automata:
    transitions = np.empty((len(dfa.states), len(dfa.states)), dtype=object)
    alphabet = []
    for symbol in dfa.alphabet.symbols:
        alphabet.append(symbol)
    
    initial_state = dfa.initial_state.name
    
    for row in range(len(dfa.states)):
        for col in range(len(dfa.states)):
            transitions[row, col] = set()
            
    final_states = set([state.name for state in dfa.states if state.is_final])
    pos_dict = {}
    for i, state in enumerate(dfa.states):
        pos_dict[state.name] = i
    
    for state in dfa.states:
        for symbol in dfa.alphabet.symbols:
            dest_state = state.transitions[symbol].pop()
            transitions[pos_dict[state.name], pos_dict[dest_state.name]].add(symbol)
        
    return Automata(transitions, final_states, alphabet, initial_state, pos_dict)

def make_automata_connected(dfa: DFA) -> DFA:
    for state in dfa.states:
        is_hole = is_hole_state(dfa, state)
        if is_hole:
            prob = 0.5
            if np.random.rand() < prob:
                state.is_final = True
            else:
                # make random transition from hole state
                random_symbol = np.random.choice(dfa.alphabet.symbols)
                random_state = np.random.choice([s for s in dfa.states if s != state])
                dfa.transitions[random_symbol] = random_state 
                 
    return dfa

def is_hole_state(dfa: DFA, state: State) -> bool:
    for symbol in dfa.alphabet.symbols:
        if state.transitions[symbol] != state:
            return False
    
    return True

def make_automata_unique_accepting(dfa: DFA) -> DFA:
    for state in dfa.states:
        if state.is_final:
            state.is_final = False
            
    # make random state final
    random_state = np.random.choice([s for s in dfa.states])
    
    for state in dfa.states:
        if state == random_state:
            state.is_final = True
            
    return dfa

def make_automata_minimal(dfa: DFA) -> DFA:
    learner = LStarFactory.get_dfa_lstar_learner()
    comparator = DFAComparisonStrategy()
    teacher = GeneralTeacher(state_machine=dfa, comparison_strategy=comparator)
    minimal_dfa = learner.learn(teacher).model
    
    for state in minimal_dfa.states:
        if state.name == 'ϵ':
            state.name = 'epsilon'
    
    return minimal_dfa