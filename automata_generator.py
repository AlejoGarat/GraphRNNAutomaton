from automata import Automata
from exporter import export_automatas
import networkx as nx

def generate_and_export_automatas(property: int, n: int, path: str):
    """Generates n automatas from a given property and exports them to a pickle file.
    
    Args:
        property (int): The property to generate the automatas from.
        n (int): The number of automatas to generate.
        path (str): The path to the pickle file.
    """
    automatas = generate_many_automatas(property, n)
    export_automatas(automatas, path)

def generate_many_automatas(property: int, n: int) -> [Automata]:
    """Generates n automatas from a given property.
    
    Args:
        property (int): The property to generate the automatas from.
        n (int): The number of automatas to generate.
        
    Returns:
        [Automata]: The generated automatas.
    """
    generated_automatas = []
    for _ in range(n):
        generated_automatas.append(generate_automata(property))
    
    return generated_automatas
    
def generate_automata(property: int) -> Automata:
    """Generates an automata from a given property.

    Args:
        property (int): The property to generate the automata from.

    Returns:
        Automata: The generated automata.
    """
    if property == "connected":
        return generate_connected_automata()
    elif property == "fully_accepting":
        return generate_fully_accepting_automata()
    elif property == "minimal":
        return generate_minimal_automata()
    
    return None
        
def generate_connected_automata():
    automaton = nx.DiGraph()

    # Definir los estados del autómata
    states = ["q0", "q1", "q2", "q3"]

    # Agregar estados como nodos al grafo
    automaton.add_nodes_from(states)

    # Definir el estado inicial
    initial_state = "q0"

    # Definir los estados finales
    final_states = ["q3"]

    # Agregar arcos (transiciones) entre estados
    automaton.add_edge("q0", "q1", label="a")
    automaton.add_edge("q1", "q2", label="b")
    automaton.add_edge("q2", "q3", label="c")

    # Verificar la conectividad del autómata
    if nx.is_strongly_connected(automaton):
        print("El autómata es conexo.")
    else:
        print("El autómata no es conexo.")

    # Para visualizar el autómata (requiere Matplotlib)
    import matplotlib.pyplot as plt
    nx.draw(automaton, with_labels=True, node_size=500, node_color="lightblue")
    plt.show()


def generate_fully_accepting_automata():
    return NotImplementedError()

def generate_minimal_automata():
    return NotImplementedError()