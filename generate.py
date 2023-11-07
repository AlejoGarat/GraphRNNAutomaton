import automata_generator

def generate_automatons_connected_property():
    # Example of how to generate automatons given the following parameters:
    # - Property: connected
    # - Amount: 10
    # - Nominal size: 5
    # - Alphabet size: 2
    # - Path: automatas.pickle
    automata_generator.generate_and_export_automatas(
        property="connected", 
        amount=1, 
        nominal_size=10, 
        alphabet_size=2, 
        path="connected_property_automatas"
    )
    
def generate_automatons_unique_accepting_property():
    # Example of how to generate automatons given the following parameters:
    # - Property: unique_accepting
    # - Amount: 10
    # - Nominal size: 5
    # - Alphabet size: 2
    # - Path: automatas.pickle
    automata_generator.generate_and_export_automatas(
        property="unique_accepting", 
        amount=1, 
        nominal_size=10, 
        alphabet_size=2, 
        path="unique_accepting_property_automatas"
    )
    
def generate_automatons_minimal_property():
    # Example of how to generate automatons given the following parameters:
    # - Property: minimal
    # - Amount: 10
    # - Nominal size: 5
    # - Alphabet size: 2
    # - Path: automatas.pickle
    automata_generator.generate_and_export_automatas(
        property="minimal", 
        amount=1, 
        nominal_size=10, 
        alphabet_size=2, 
        path="minimal_property_automatas"
    )
    
generate_automatons_minimal_property()