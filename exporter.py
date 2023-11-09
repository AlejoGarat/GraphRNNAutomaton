from automata import Automata
import pickle 
import numpy as np

def export_automatas(automatas: [Automata], path: str):
    """Exports the automatas to a pickle file.

    Args:
        automatas ([Automata]): The automatas to export.
    """
    pkl_file = open(f'{path}.pkl', 'wb')
    automata_info = bfs_processed_automata_info(automatas)
    pickle.dump(automata_info, pkl_file)
    pkl_file.close()
    
def bfs_processed_automata_info(automatas: [Automata]) -> dict():
    bfs_automatas_info = []
    
    for automata in automatas:
        queue = []
        node_dict = {}
        initial_node = automata.pos_dict[automata.initial_state]
        
        order = -1
        queue.append(initial_node)
                
        while len(queue) > 0:
            order += 1
            node = queue.pop(0)
            node_dict[node] = order
            for neighbor, connected_set in enumerate(automata.transitions[node]):
                if connected_set and neighbor not in node_dict and neighbor not in queue:
                    queue.append(neighbor)
                    
        new_transitions = np.zeros((len(automata.transitions), len(automata.transitions)), dtype=object)
        for row in range(len(automata.transitions)):
            for col in range(len(automata.transitions)):
                new_row = node_dict[row]
                new_col = node_dict[col]
                
                if automata.transitions[new_row, new_col]:
                    new_transitions[row, col] = 1
                else:
                    new_transitions[row, col] = 0 
                
        new_final_states = set()
        for state in automata.final_states:
            new_final_states.add(node_dict[automata.pos_dict[state]])
            
        automata_info = {"transitions": new_transitions, "final_states": new_final_states, 
                      "initial_state": automata.pos_dict[automata.initial_state]}
        bfs_automatas_info.append(automata_info)
        
    return bfs_automatas_info
                

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