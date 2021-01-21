import threading
import numpy as np


class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        #inc is the number of samples being included
        inc = inc or 1
        # if there is enough space in the buffer
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)

        # if there is not enough space in the buffer
        elif self.current_size < self.size:
            # Calculate the overflow
            overflow = inc - (self.size - self.current_size)

            # Allocate the remaining buffer space
            idx_a = np.arange(self.current_size, self.size)
            # Rnadomly select from the entire buffer space indices 
            idx_b = np.random.randint(0, self.current_size, overflow)
            
            # Concatenate the two to retrieve the final index listing
            idx = np.concatenate([idx_a, idx_b])
        else:
            # If the buffer is full already, select any index from the buffer
            idx = np.random.randint(0, self.size, inc)

        # Update the current size 
        self.current_size = min(self.size, self.current_size+inc)

        # Special case?
        if inc == 1:
            idx = idx[0]
        
        
        return idx
