import numpy as np


if __name__ == "__main__":

        # Load the data
        data = np.load('./data/dataset.npy', allow_pickle=True)

        embeddings_dataset = {}

        # Get the unique states and actions
        states = np.unique(data[:,0].ravel().tolist(),axis=0)
        actions = np.unique(data[:,1].ravel().tolist(),axis=0) 

        state_to_id = {}
        id_to_state = {}
        state_to_id[0] = "UNK"
        id_to_state["UNK"] = 0
        id = 1
        for state in states:
                state_to_id[tuple(state)] = id
                id_to_state[id] = state
                id += 1

        action_to_id = {}
        id_to_action = {}
        action_to_id["UNK"] = 0
        id_to_action[0] = "UNK"
        id = 1
        for action in actions:
                action_to_id[tuple(action)] = id
                id_to_action[id] = action
                id += 1

        
        embeddings_dataset['state_to_id'] = state_to_id
        embeddings_dataset['id_to_state'] = id_to_state
        embeddings_dataset['action_to_id'] = action_to_id
        embeddings_dataset['id_to_action'] = id_to_action

        embeddings_dataset['state_features'] = data[:,0].ravel()
        embeddings_dataset['action_features'] = data[:,1].ravel()
        embeddings_dataset['targets'] = data[:,2].ravel()

        np.save('./data/embeddings_dataset', embeddings_dataset)