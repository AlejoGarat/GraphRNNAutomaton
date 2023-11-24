import wandb

def get_wandb_sweep_name(property_name, state_amount, alphabet_size):
    if property_name == 'minimal':
        return f'Minimal DFA S{state_amount} A{alphabet_size}'
    if property_name == 'unique_accepting':
        return f'Unique Accepting State DFA S{state_amount} A{alphabet_size}'
    if property_name == 'connected':
        return f'Fully Connected DFA S{state_amount} A{alphabet_size}'
    else:
        raise NameError('Incorrect Property!')
    
def create_sweep(name, parameters, method, project, entity):
    sweep_config = {
        'method': method,
        'name':name,
        'metric': {
            'goal':'maximize', 'name':'accuracy'
        },
        'parameters':parameters
    }

    return wandb.sweep(sweep=sweep_config, project=project, entity=entity)