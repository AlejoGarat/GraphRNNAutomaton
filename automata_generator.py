from automata import Automata
from exporter import export_automatas

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
    if property == "property_1":
        return generate_automata_property_1()
    elif property == 2:
        return generate_automata_property_2()
    elif property == 3:
        return generate_automata_property_2()
    
    return None
        
def generate_automata_property_1():
    return NotImplementedError()

def generate_automata_property_2():
    return NotImplementedError()

def generate_automata_property_3():
    return NotImplementedError()