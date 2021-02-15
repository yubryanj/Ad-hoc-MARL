import numpy as np
import random

def transition_function(initial_states, joint_actions, args):

    # Note: Currently multiple agents can occupy a cell

    # Allocate storage
    next_state = []

    # For each agent
    for initial_state, action in zip(initial_states, joint_actions):

        # Compute the next position
        next_row = min(max(initial_state[0]+action[0],0),args['n_rows']-1)
        next_col = min(max(initial_state[1]+action[1],0),args['n_cols']-1)
        next_position = ( next_row, next_col)

        # Add it to the next state
        next_state.append(next_position)

    return next_state

def generate(args):

    # Sample a set of initial states
    initial_states = [np.random.choice(len(args['states']), args['n_agents'], replace=False) for _ in range(int(args['n_samples']))]
    initial_states = [[args['states'][i] for i in positions] for positions in initial_states]

    # sample a set of actions
    joint_actions = [np.random.choice(len(args['actions']), args['n_agents'], replace=True) for _ in range(int(args['n_samples']))]
    joint_actions = [[args['actions'][i] for i in joint_action] for joint_action in joint_actions]

    # Compute next state given initial state and joint action
    next_states = [transition_function(initial_state,joint_action, args) for initial_state, joint_action in zip(initial_states, joint_actions)]

    # Convert to string
    initial_states_names = [[args['state_to_state_name'][state] for state in initial_state] for initial_state in initial_states]
    joint_actions_names = [[args['action_to_action_name'][action] for action in joint_action] for joint_action in joint_actions]
    next_state_names = [[args['state_to_state_name'][state] for state in next_state] for next_state in next_states]

    features =[]
    for initial_state_name, joint_action in zip(initial_states_names, joint_actions_names):
        features.append(np.hstack((initial_state_name, joint_action)))

    features = np.array(features)
    targets = np.array(next_state_names)


    return features, targets


if __name__=="__main__":

    args ={ 'n_rows': 3,
            'n_cols': 3,
            'n_agents': 2,
            'n_samples': 1e3,
    }

    # Define the state and action space
    args['states'] = [(row,col) for row in range(args['n_rows']) for col in range(args['n_cols'])]
    args['actions'] = [(0,0),(0,1),(0,-1),(1,0),(-1,0)]
    args['state_names'] = [f's{row}{col}' for row in range(args['n_rows']) for col in range(args['n_cols'])]
    args['action_names'] = ['stay','up','down','right','left']
    args['state_to_state_name'] = {}
    args['action_to_action_name'] = {}
    args['id_to_word'] = {}
    args['word_to_id'] = {}
    args['maximum_sequence_length'] = args['n_agents'] * 2 + 2

    # Generate a mapping from word to id and vice versa
    
    args['id_to_word'][0] = 'SOS'
    args['id_to_word'][1] = 'EOS'
    args['id_to_word'][2] = '<PAD>'
    args['word_to_id']['SOS'] = 0
    args['word_to_id']['EOS'] = 1
    args['word_to_id']['<PAD>'] = 2

    id = 3
    for state, state_name in zip(args['states'],args['state_names']):
        args['state_to_state_name'][state] = state_name
        args['id_to_word'][id] = state_name
        args['word_to_id'][state_name] = id
        id += 1

    for action, action_name in zip(args['actions'],args['action_names']):
        args['action_to_action_name'][action] = action_name 
        args['id_to_word'][id] = action_name
        args['word_to_id'][action_name] = id
        id += 1

    args['number_of_words'] = id + 1

    # Generate the dataset
    features, targets = generate(args)

    # Retrieve only the unique set
    features, indices = np.unique(features,axis=0, return_index=True)
    targets = targets[indices]

    np.save('./data/features.npy',features)
    np.save('./data/target.npy',targets)
    np.save('./data/args.npy',args)