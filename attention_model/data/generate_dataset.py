import numpy as np



def prepare_data(dataset, max_number_of_agents=6, generate_sample_size=int(1e5)):
        n_samples = len(dataset['state_features'])
        dataset['action_to_id'][(0,0,0,0,0)] = 6
        states = []
        actions = []
        targets = []

        for _ in range(generate_sample_size):
                number_of_agents = np.random.choice([1,2,3,5,6])
                samples = np.random.choice(n_samples, size=number_of_agents)

                # Select and pad
                state = np.array([dataset['state_features'][i] for i in samples] + [(0,0) for i in range(max_number_of_agents - number_of_agents)]) 
                action = np.array([dataset['action_features'][i] for i in samples] + [(0,0,0,0,0) for i in range(max_number_of_agents - number_of_agents)])
                target = np.array([dataset['targets'][i] for i in samples] + [(0,0) for i in range(max_number_of_agents - number_of_agents)])

                # Collect the entries
                states.append(state)
                actions.append(action)
                targets.append(target)

        data = {'state_features': states,
                'action_features':actions,
                'targets':targets,
                'state_to_id':dataset['state_to_id'],
                'action_to_id':dataset['action_to_id']}

        return data


if __name__ == "__main__":
    
        dataset = np.load('./data/embeddings_dataset.npy', allow_pickle=True).item()
        training_dataset = prepare_data(dataset)

        np.save('./data/embeddings_dataset_variable_agents.npy',training_dataset)

        validation_dataset = prepare_data(dataset, max_number_of_agents=6, generate_sample_size=int(1000))
        np.save('./data/validation.npy', validation_dataset)
