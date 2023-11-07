class Automata():
    def __init__(self, transitions, final_states: set = {}, alphabet: [] = [], initial_state: str = '', pos_dict: dict = {}):
        self.transitions = transitions
        self.final_states = final_states
        self.alphabet = alphabet
        self.initial_state = initial_state
        self.pos_dict = pos_dict
        
    def __str__(self):
        return f"Transitions: {self.transitions}\nFinal states: {self.final_states}"
    
    def get_transitions(self, state: int):
        in_transitions = self.transitions[:,state]
        loop_transition = self.transitions[state,state]
        out_transitions = self.transitions[state,:]
        return in_transitions, loop_transition, out_transitions
    
    def is_final_state(self, state: int):
        return state in self.final_states
    
    def has_transitions(self, state: int):
        in_t, loop_t, out_t = self.get_transitions(state)
        return len(in_t) > 0 or len(loop_t) > 0 or len(out_t) > 0
        
        