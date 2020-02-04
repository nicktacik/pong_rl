#!/usr/bin/env python
# coding: utf-8

# # Talko Tuesday 
# # February 4th, 2020
# # How to become a pong master with OpenAI Gym
# 
# 
# 
# 

# ## Introduction to Gym

# In[1]:


import gym
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython import display


# In[2]:


from gym import envs
for env in envs.registry.all():
    print(env)


# In[3]:


env = gym.make('PongNoFrameskip-v4')
observation = env.reset()
for i in range(1000):
    if i % 10 == 0:
        plt.imshow(env.render(mode='rgb_array')) 
        display.display(plt.gcf())
        display.clear_output(wait=True)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break


# In[4]:


print(observation)


# In[5]:


observation.shape


# In[6]:


env.action_space


# In[7]:


reward


# ## Let's begin building our bot

# In[8]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam


# In[9]:


class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)
    
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, new_states, dones


# In[10]:


def build_dqn(lr, n_actions, input_dims):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', 
                    input_shape=(*input_dims,), data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                     data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                    data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions))
    
    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')
    
    return model


# In[11]:


class Agent():
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, replace, 
                 input_dims, action_map, eps_dec=1e-5, eps_min=0.01, mem_size=1e6, 
                 q_eval_fname='q_eval', q_target_fname='q_target'):
        self.action_space = [i for i in range(n_actions)]
        self.action_map = action_map
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.replace = replace
        self.q_eval_fname = q_eval_fname
        self.q_target_fname = q_target_fname
        self.learn_step = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(alpha, n_actions, input_dims)
        self.q_next = build_dqn(alpha, n_actions, input_dims)
        self.q_eval_model_file = q_eval_fname
        self.q_target_model_file = q_target_fname
    
    def replace_target_network(self):
        if self.replace and self.learn_step % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())
    
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def choose_action(self, observation):
        if np.random.rand() < self.epsilon:
            action =np.random.choice(self.action_space)
        else:
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return self.action_map[action], action
    
    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            states, actions, rewards, new_states, done = self.memory.sample_buffer(self.batch_size)
            self.replace_target_network()
            q_eval = self.q_eval.predict(states)
            q_next = self.q_next.predict(new_states)
            
            q_next[done] = 0.0
            
            indices = np.arange(self.batch_size)
            q_target = q_eval[:]
            
            q_target[indices, actions] = reward + self.gamma * np.max(q_next, axis=1)
            
            self.q_eval.train_on_batch(states, q_target)
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
            self.learn_step += 1
    
    def save_models(self, extention=""):
        self.q_eval.save(self.q_eval_model_file + extention + ".h5")
        self.q_next.save(self.q_target_model_file + extention + ".h5")
    
    def load_models(self, extention=""):
        self.q_eval = load_model(self.q_eval_model_file + extention + ".h5")
        self.q_next = load_model(self.q_target_model_file + extention + ".h5")


# In[12]:


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip
    
    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info


# In[13]:


class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(80, 80, 1),
                                               dtype=np.uint8)
    
    def observation(self, obs):
        return PreProcessFrame.process(obs)
    
    @staticmethod
    def process(frame):
        new_frame = np.reshape(frame, frame.shape).astype(np.float32)
        new_frame = 0.299*new_frame[:, :, 0] + 0.587*new_frame[:, :, 1] + 0.114*new_frame[:, :, 2]
        new_frame = new_frame[35:195:2, ::2].reshape(80, 80, 1)
        return new_frame.astype(np.uint8)


# In[14]:


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


# In[15]:


class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape = (self.observation_space.shape[-1],
                     self.observation_space.shape[0],
                     self.observation_space.shape[1],            
            ),
            dtype=np.float32
        )
    
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


# In[16]:


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(n_steps, axis=0),
            env.observation_space.high.repeat(n_steps, axis=0),
            dtype=np.float32
        )
    
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())
    
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


# In[17]:


def make_env(env_name):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)


# In[18]:


env = make_env('PongNoFrameskip-v4')
num_games = 20000
load_checkpoint = False
best_score = -21
agent = Agent(gamma=0.99, epsilon=1.0, alpha=1e-4, input_dims=(4,80,80),
              action_map={0:0, 1:4, 2:5},
              n_actions=3, mem_size=25000, eps_min=0.02, batch_size=32,
              replace=1000, eps_dec=1e-5)


# In[19]:


scores, eps_history = [], []
n_steps = 0

for i in range(num_games):
    score = 0
    observation = env.reset()
    done = False
    while not done:
        action, action_id = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward
        agent.store_transition(observation, action_id, reward, observation_, int(done))
        agent.learn()
        observation = observation_
    scores.append(score)
    
    avg_score = np.mean(scores[-100:])
    print(f"episode: {i}, avg_score: {avg_score}")
    
    if avg_score > best_score:
        agent.save_models()
        best_score = avg_score
        
    if num_games % 100 == 0:
        agent.save_models(extention = str(i))
        


# In[20]:


get_ipython().system('pip install pandoc')


# In[ ]:





# In[ ]:





# In[ ]:




