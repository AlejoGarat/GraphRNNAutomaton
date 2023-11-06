import numpy as np
from automata import Automata
from exporter import export_automatas
from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton as DFA
from pythautomata.base_types.state import State
from pythautomata.base_types.symbol import Symbol
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.utilities.nicaud_dfa_generator import generate_dfa
from pymodelextractor.learners.observation_table_learners.general_lstar_learner import GeneralLearner

def generate_and_export_automatas(property: str, amount: int, nominal_size: int, alphabet_size: int, path: str):
    """Generates n automatas from a given property and exports them to a pickle file.
    
    Args:
        property (int): The property to generate the automatas from.
        n (int): The number of automatas to generate.
        path (str): The path to the pickle file.
    """
    automatas = generate_many_automatas(amount, nominal_size, alphabet_size)
    if property == "connected":
        automatas = [pythautomata_automata_to_automata(make_automata_connected(automata)) for automata in automatas]
    elif property == "fully_accepting":
        automatas = [pythautomata_automata_to_automata(make_automata_unique_accepting(automata)) for automata in automatas]
    elif property == "minimal":
        automatas = [pythautomata_automata_to_automata(make_automata_minimal(automata)) for automata in automatas]
    else:
        raise ValueError("Unknown property")

    export_automatas(automatas, path)

def generate_many_automatas(amount: int, nominal_size: int, alphabet_size: int) -> [DFA]:
    """Generates n automatas using Nicaud.
    
    Args:
        property (int): The property to generate the automatas from.
        n (int): The number of automatas to generate.
        
    Returns:
        [Automata]: The generated automatas.
    """
    alphabet = lambda: Alphabet([Symbol(str(i)) for i in range(alphabet_size)])
    generated_automatas = []
    for _ in range(amount):
        generated_automatas = generate_dfa(alphabet=alphabet, nominal_size=nominal_size)
    
    return generated_automatas

def pythautomata_automata_to_automata(dfa: DFA) -> Automata:
    """Converts a pythautomata DFA to an Automata.
    
    Args:
        dfa (DFA): The DFA to convert.
        
    Returns:
        Automata: The converted automata.
    """
    transitions = np.zeros((len(dfa.states), len(dfa.states)))
    final_states = dfa.final_states
    for state in dfa.states:
        for symbol in dfa.alphabet:
            transitions[state.name, dfa.transitions[state, symbol]].append(symbol.name)
        
    return Automata(transitions, final_states)
    
def make_automata_connected(dfa: DFA) -> DFA:
    for state in dfa.states:
        is_hole = is_hole_state(dfa, state)
        if is_hole:
            prob = 0.5
            if np.random.rand() < prob:
                state.is_final = True
            else:
                # make random transition from hole state
                random_symbol = np.random.choice(dfa.alphabet)
                dfa.transitions[random_symbol, state] = np.random.choice(dfa.alphabet)

def is_hole_state(dfa: DFA, state: State) -> bool:
    for symbol in dfa.alphabet:
        if dfa.transitions[state, symbol] != state:
            return False
    
    return True

def make_automata_unique_accepting(dfa: DFA) -> DFA:
    for state in dfa.states:
        if dfa.is_final_state(state):
            state.is_final = False
            
    # make random state final
    dfa_size = len(dfa.states)
    random_state = np.random.randint(dfa_size)
    dfa.states[random_state].is_final = True
    
    return dfa

def make_automata_minimal(dfa: DFA) -> DFA:
    learner = GeneralLearner(dfa)
    minimal_dfa = learner.learn()
    return minimal_dfa