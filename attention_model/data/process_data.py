import numpy as np

def prepare_data(dataset):
        features = []
        targets = []
        for sample in dataset:
                initial_state = np.concatenate(sample[0]).ravel()
                action = np.concatenate(sample[1]).ravel()
                next_state = np.concatenate(sample[2]).ravel()

                features.append(np.concatenate((initial_state,action)))
                targets.append(next_state)

        return features, targets


if __name__ == "__main__":
    
    dataset = np.load('./data/dataset.npy', allow_pickle=True)
    features, targets = prepare_data(dataset)

    np.save('./data/features',features)
    np.save('./data/targets',targets)