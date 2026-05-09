import os
import torch as T
import numpy as np
import torch.nn.functional as F
from networks import *
from buffer import ReplayBuffer

class Agent():
    def __init__(self,actor_learning_rate, critic_learning_rate, input_dims, tau, env, gamma=0.99, update_actor_interval=2,warmup=5000,
                 n_actions=2,max_size=1000000, layer1_size=256, layer2_size=256,
                 batch_size=100,noise=0.1):
        self.gamma=gamma
        self.tau = tau
        self.max_action=env.action_space.high
        self.min_action=env.action_space.low
        self.memory = ReplayBuffer(max_size,input_dims,n_actions)
        self.batch_size=batch_size
        self.learn_step_count=0
        self.time_step=0
        self.warmup=warmup
        self.n_actions=n_actions
        self.update_actor_interval= update_actor_interval


        self.actor = ActorNetwork(input_dims=input_dims,fc1_dims=layer1_size,
                                  fc2_dims=layer2_size,n_actions=n_actions,name="actor",learning_rate=actor_learning_rate)
        self.critic_1=CriticNetwork(input_dims=input_dims,fc1_dims=layer1_size,
                                  fc2_dims=layer2_size,n_actions=n_actions,name="citic_1",learning_rate=critic_learning_rate)
        self.critic_2=CriticNetwork(input_dims=input_dims,fc1_dims=layer1_size,
                                  fc2_dims=layer2_size,n_actions=n_actions,name="citic_2",learning_rate=critic_learning_rate)
    
        self.target_actor = ActorNetwork(input_dims=input_dims,fc1_dims=layer1_size,
                                  fc2_dims=layer2_size,n_actions=n_actions,name="target_actor",learning_rate=actor_learning_rate)
        self.target_critic_1=CriticNetwork(input_dims=input_dims,fc1_dims=layer1_size,
                                  fc2_dims=layer2_size,n_actions=n_actions,name="citic_1",learning_rate=critic_learning_rate)
        self.target_critic_2=CriticNetwork(input_dims=input_dims,fc1_dims=layer1_size,
                                  fc2_dims=layer2_size,n_actions=n_actions,name="citic_2",learning_rate=critic_learning_rate)
    
        self.noise=noise
        self.update_network_paramters(tau=1)
    def choose_action(self,observation,validation=False):
        if self.time_step < self.warmup and validation is False:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)), dtype=T.float).to(self.actor.device)
            mu = T.clamp(mu, T.tensor(self.min_action[0], dtype=T.float).to(self.actor.device), T.tensor(self.max_action[0], dtype=T.float).to(self.actor.device))
        else:
            state=T.tensor(observation,dtype=T.float).to(self.actor.device)
            mu =self.actor.forward(state).to(self.actor.device)

        mu_prime= mu + T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)), dtype=T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, T.tensor(self.min_action[0], dtype=T.float).to(self.actor.device), T.tensor(self.max_action[0], dtype=T.float).to(self.actor.device)) 
             
        self.time_step+=1
        return mu_prime.cpu().detach().numpy()
        
        


    def remember_transition(self,state,action,reward,next_state,done):
        self.memory.store_transition(state,action,reward,next_state,done)
        
        
    def learn(self):
        if self.memory.mem_cntr<self.batch_size*10:
            return
        self.critic_loss_history = []
        self.actor_loss_history = []
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        reward=T.tensor(reward,dtype=T.float).to(self.actor.device)
        state=T.tensor(state,dtype=T.float).to(self.actor.device)
        action=T.tensor(action,dtype=T.float).to(self.actor.device)
        next_state=T.tensor(next_state,dtype=T.float).to(self.actor.device)
        done=T.tensor(done,dtype=T.bool).to(self.actor.device)
        target_actions = self.target_actor.forward(next_state).to(self.actor.device)
        target_actions=target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2, size=(self.batch_size, self.n_actions)), dtype=T.float).to(self.actor.device),-0.5,0.5)
        target_actions=T.clamp(target_actions,self.min_action[0],self.max_action[0])

        qnext_1=self.target_critic_1.forward(next_state,target_actions).to(self.actor.device)
        qnext_2=self.target_critic_2.forward(next_state,target_actions).to(self.actor.device)
        q1=self.critic_1.forward(state,action).to(self.actor.device)
        q2=self.critic_2.forward(state,action).to(self.actor.device)
        qnext_1[done]=0.0
        qnext_2[done]=0.0
        qnext_1=qnext_1.view(-1)
        qnext_2=qnext_2.view(-1)
        q1=q1.view(-1)
        q2=q2.view(-1)
        next_critic_value=T.min(qnext_1,qnext_2)
        target=reward + self.gamma*next_critic_value
        target=target.view(-1)

        q1_loss=F.mse_loss(q1,target)
        q2_loss=F.mse_loss(q2,target)

        critic_loss=q1_loss+q2_loss
        self.last_critic_loss = critic_loss.item()
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        critic_loss.backward()

 
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.learn_step_count+=1
        
        
        if self.learn_step_count % self.update_actor_interval !=0:
            return
            
        self.actor.optimizer.zero_grad()
        actor_actions = self.actor.forward(state)
        actor_q1_loss = self.critic_1.forward(state, actor_actions)
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        self.last_actor_loss = actor_loss.item()
        self.update_network_paramters(tau=self.tau)
        
        
        


    def update_network_paramters(self,tau=None):
        if tau==None:
            tau=self.tau
        actor_param=self.actor.named_parameters()
        critic_1_param = self.critic_1.named_parameters()
        critic_2_param = self.critic_2.named_parameters()
        target_actor_param = self.target_actor.named_parameters()
        target_critic_1_param = self.target_critic_1.named_parameters()
        target_critic_2_param = self.target_critic_2.named_parameters()
        
        actor_state_dict = dict(actor_param)
        critic_1_state_dict = dict(critic_1_param)
        critic_2_state_dict = dict(critic_2_param)
        target_actor_state_dict = dict(target_actor_param)
        target_critic_1_state_dict = dict(target_critic_1_param)
        target_critic_2_state_dict = dict(target_critic_2_param)
        
        for name in target_critic_1_state_dict:
            target_critic_1_state_dict[name]=tau*critic_1_state_dict[name].clone() + (1-tau)*target_critic_1_state_dict[name].clone()
        for name in target_critic_2_state_dict:
            target_critic_2_state_dict[name]=tau*critic_2_state_dict[name].clone() + (1-tau)*target_critic_2_state_dict[name].clone()
        for name in target_actor_state_dict:
            target_actor_state_dict[name]=tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()
        
        self.target_actor.load_state_dict(target_actor_state_dict)
        self.target_critic_1.load_state_dict(target_critic_1_state_dict)
        self.target_critic_2.load_state_dict(target_critic_2_state_dict)
        
        
        


    def save_model(self):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
    def load_model(self):

        try:
            self.actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.target_critic_1.load_checkpoint()
            self.target_critic_2.load_checkpoint()
            
        except Exception as e:
            print(e)

        




    
