from automata import Automata
import pickle 

def export_automatas(automatas: [Automata], path: str):
    """Exports the automatas to a pickle file.

    Args:
        automatas ([Automata]): The automatas to export.
    """
    pkl_file = open(f'{path}.pkl', 'wb')
    transitions_final_states = [(automata.transitions, automata.final_states) for automata in automatas]
    pickle.dump(transitions_final_states, pkl_file)
    pkl_file.close()


def read_automatas(path: str) -> [Automata]:
    """Reads the automatas from a pickle file.

    Args:
        path (str): The path to the pickle file.

    Returns:
        [Automata]: The automatas read from the pickle file.
    """
    pkl_file = open(f'{path}.pkl', 'rb')
    automatas = pickle.load(pkl_file)
    pkl_file.close()
    return automatas