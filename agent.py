from replay_buffer import ReplayBuffer
from generate_dqn import generate_dqn
from keras.models import load_model
import numpy as np


class Agent:

    def __init__(self, alpha, gamma, n_actions, action_map, mem_size, batch_size, replace, input_dims,
                 epsilon, epsilon_dec, epsilon_min,
                 q_eval_fname='./models/q_eval', q_target_fname='./models/q_target',
                 load_from_checkpoint=False):

        """
        :param alpha: Model learning rate
        :param gamma: Discount factor
        :param n_actions: Size of the action space
        :param action_map: A dictionary that maps from the action space to actions
        :param mem_size: The maximum memory of the replay buffer
        :param batch_size: Size of batches to train on
        :param replace: How often update target network
        :param input_dims: Structure of the tensor passed into DQN
        :param epsilon: How often to act randomly
        :param epsilon_dec: How much to decrement epsilon by every step
        :param epsilon_min: Minimum value of epsilon
        :param q_eval_fname: Base file name for saving eval network
        :param q_target_fname: Base file name for saving target network
        """

        self.gamma = gamma

        self.action_space = [i for i in range(n_actions)]
        self.action_map = action_map

        self.batch_size = batch_size
        self.replace = replace

        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min

        self.q_eval_fname = q_eval_fname
        self.q_target_fname = q_target_fname

        self.learn_step = 0
        self.memory = ReplayBuffer(mem_size, input_dims)

        if load_from_checkpoint:
            self.load_models()
        else:
            self.q_eval = generate_dqn(alpha, n_actions, input_dims)
            self.q_next = generate_dqn(alpha, n_actions, input_dims)

    def save_models(self, extension=""):
        self.q_eval.save(f"{self.q_eval_fname}{extension}.h5")
        self.q_next.save(f"{self.q_target_fname}{extension}.h5")

    def load_models(self, extension=""):
        self.q_eval = load_model(f"{self.q_eval_fname}{extension}.h5")
        self.q_next = load_model(f"{self.q_target_fname}{extension}.h5")

    def replace_target_network(self):
        if self.replace and self.learn_step % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def get_action(self, observation):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_count <= self.batch_size:
            return

        states, actions, rewards, new_states, done = self.memory.sample_buffer(self.batch_size)
        self.replace_target_network()
        q_eval = self.q_eval.predict(states)
        q_next = self.q_next.predict(new_states)

        q_next[done] = 0.0

        indices = np.arange(self.batch_size)

        q_target = q_eval[:]
        q_target[indices, actions] = rewards + self.gamma * np.max(q_next, axis=1)

        self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
        self.learn_step += 1

