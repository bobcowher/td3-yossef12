from turtle import done

import numpy as np


class ReplayBuffer():
    def __init__(self,max_size, input_shape,n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        input_shape_tuple = (input_shape,) if isinstance(input_shape, int) else tuple(input_shape)
        self.state_memory = np.zeros((self.mem_size, *input_shape_tuple))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape_tuple))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.done_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index]=done

        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch=np.random.choice(max_mem,batch_size)

        state=self.state_memory[batch]
        next_state=self.new_state_memory[batch]
        action=self.action_memory[batch]
        reward=self.reward_memory[batch]
        done=self.done_memory[batch]
        return state,action,reward,next_state,done
    

       
