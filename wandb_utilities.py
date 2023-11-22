import wandb

def get_wandb_sweep_name(property_name):
    if property_name == 'minimal':
        return 'Minimal DFA'
    if property_name == 'unique_accepting':
        return 'Unique Accepting State DFA'
    if property_name == 'connected':
        return 'Fully Connected DFA'
    else:
        raise NameError('Incorrect Property!')
    
def create_sweep(name, parameters, method, project, entity):
    sweep_config = {
        'method': method,
        'name':name,
        'metric': {
            'goal':'maximize', 'name':'val_acc'
        },
        'parameters':parameters
    }

    return wandb.sweep(sweep=sweep_config, project=project, entity=entity)