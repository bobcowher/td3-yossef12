import sys
sys.modules['coverage'] = None
import os
import torch
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import warnings
import logging

# Suppress robosuite controller warnings
warnings.filterwarnings('ignore', category=UserWarning, module='robosuite')
logging.getLogger('robosuite').setLevel(logging.ERROR)

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from networks import *
from buffer import ReplayBuffer
from td3 import Agent
if __name__ == "__main__":

    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    controller_config = load_composite_controller_config(
        controller="BASIC"
    )

    env = suite.make(
        env_name="Door",
        robots="Panda",
        controller_configs=controller_config,
        use_camera_obs=False,
        has_renderer=False,
        horizon=300,
        render_camera="frontview",
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20
    )

    env = GymWrapper(env)

    actor_learning_rate = 3e-4
    critic_learning_rate = 3e-4
    tau = 0.01
    gamma = 0.99
    update_actor_interval = 2
    warmup = 10000
    n_actions = env.action_space.shape[0]
    max_size = 1000000
    layer1_size = 400
    layer2_size = 300
    batch_size = 64
    noise = 0.2
    
    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, input_dims=env.observation_space.shape[0],tau=tau, env=env, 
                gamma=gamma,update_actor_interval=update_actor_interval, warmup=warmup, n_actions=n_actions, max_size=max_size, layer1_size=layer1_size, layer2_size=layer2_size,
                batch_size=batch_size, noise=noise)

    writer=SummaryWriter("logs")
    n_games=10000
    best_score = 0
    episode_identifier=f"0 - actor_lr{actor_learning_rate}-critic_lr{critic_learning_rate}-tau{tau}-gamma{gamma}-update_actor_interval{update_actor_interval}-warmup{warmup}-n_actions{n_actions}-max_size{max_size}-layer1_size{layer1_size}-layer2_size{layer2_size}-batch_size{batch_size}-noise{noise}"
    
    agent.load_model()
    for i in range(n_games):
        observation,_=env.reset()
        done = False
        score=0
        while not done:
            action=agent.choose_action(observation)
            next_observation,reward,done,truncated,info=env.step(action)
            score+=reward
            agent.remember_transition(observation,action,reward,next_observation,done)
            observation=next_observation 
        for _ in range(10):
            agent.learn()
        writer.add_scalar(f"score{episode_identifier}",score,global_step=i)

        if (i%10)==0:
            agent.save_model()
            
            
        print(f"episode {i} score{score}")











    obs = env.reset()
