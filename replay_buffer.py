import numpy as np


class ReplayBuffer:

    def __init__(self, max_size, input_shape):
        self.max_size = max_size
        self.mem_count = 0
        self.states = np.zeros((self.max_size, *input_shape), dtype=np.float32)
        self.new_states = np.zeros((self.max_size, *input_shape), dtype=np.float32)
        self.actions = np.zeros(self.max_size, dtype=np.int32)  # store actions as ints not 1-hot
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_count % self.max_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index] = new_state
        self.dones[index] = done
        self.mem_count += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_count, self.max_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        new_states = self.new_states[batch]
        dones = self.dones[batch]

        return states, actions, rewards, new_states, dones
