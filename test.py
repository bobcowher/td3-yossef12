import sys
sys.modules['coverage'] = None
import os
import time
import gymnasium as gym
import numpy as np
import torch as T
import robosuite as suite
from robosuite.wrappers import GymWrapper
from torch.utils.tensorboard import SummaryWriter
from networks import *
from buffer import ReplayBuffer
from td3 import Agent
from robosuite import load_composite_controller_config
controller_config = load_composite_controller_config(
        controller="BASIC"
    )


if __name__=="__main__":
    
    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")



    env_name="Door"




    env =suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        
        use_camera_obs=False,
        has_renderer=True,

        horizon=300,
        render_camera="frontview",
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20
    )


    env=GymWrapper(env)
    actor_learning_rate = 1e-3
    critic_learning_rate = 1e-3
    tau = 0.005
    gamma = 0.99
    update_actor_interval = 2
    warmup = 1000
    n_actions = env.action_space.shape[0]
    max_size = 1000000
    layer1_size = 256
    layer2_size = 128
    batch_size = 128
    noise = 0.1
    
    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, input_dims=env.observation_space.shape[0],tau=tau, env=env, 
                gamma=gamma,update_actor_interval=update_actor_interval, warmup=warmup, n_actions=n_actions, max_size=max_size, layer1_size=layer1_size, layer2_size=layer2_size,
                batch_size=batch_size, noise=noise)
    
    n_games=3
    best_score = 0
    episode_identifier=f"0 - actor_lr{actor_learning_rate}-critic_lr{critic_learning_rate}-tau{tau}-gamma{gamma}-update_actor_interval{update_actor_interval}-warmup{warmup}-n_actions{n_actions}-max_size{max_size}-layer1_size{layer1_size}-layer2_size{layer2_size}-batch_size{batch_size}-noise{noise}"
    
    agent.load_model()
    for i in range(n_games):
        observation,_=env.reset()
        done = False
        score=0
        while not done:
            action=agent.choose_action(observation,validation=True)
            next_observation,reward,done,truncated,info=env.step(action)
            env.render()
            score+=reward
            observation=next_observation 
            time.sleep(0.03)
            
        print(f"episode {i} score{score}")

