# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:12:01 2020

@author: fabio
"""
import gym
import numpy as np
from collections import deque
from gym import spaces
import math
import os
import time 
import numpy as np
import matplotlib.pyplot as plt
import math 
from scipy.integrate import odeint, solve_ivp
pi = math.pi

'''
    Ex lecture March
'''
class system(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, action_min = -10, action_max=10, n_actions = 1, n_states = 2):
      
    # System parameters
    self.k = 2
    self.c = 0.3
    self.a = 1
    self.b = -1
    self.m = 1
    
    self.cont = 0

    # Initial conditions
    self.u = np.array([4])
    self.u1  = np.array([0])                              
    self.t = np.linspace(0,30,10000,endpoint=True)
    
    self.reset_cont = 0                                 # Iteration counter reset
    self.action_replay = deque(maxlen=4)                # Action dynamic vector
    
    # Define the upper and lower action bounds
    self.action_max = action_max
    self.action_min = action_min
    
    # Limits of displacement
    self.u_lim = 10

    # Action space definition
    self.action_space = spaces.Box(low=-1, high=1,shape=(1,), dtype=np.float32)

    # Observation space definition
    self.observation_space = spaces.Box(low=-self.u_lim, high=self.u_lim, shape=(n_states,), dtype=np.float32)

  def step(self, action):

    
    #--------------
    # Agent action scaling and action vector creation
    #--------------
    action_new = (action+1)/(2)           # scaling action from -1/1 to 0/1
    
    self.action_replay.append(action_new) # adding scaled action to the dynamic vector
    action_vec = ((self.action_max - self.action_min)*(np.mean(self.action_replay)) + self.action_min)
    
    action_vec = action
    #--------------
    
    # Forcing vector construction

    #--------------
    # Advance simulation one time step
    #--------------
    def model(t,z,u):
        dy_1dt = z[1]
        dy_2dt = -(self.c/self.m)*(self.a*z[0]**2 + self.b)*z[1] -(self.k/self.m)*z[0] + u/self.m
        return [dy_1dt,dy_2dt]
    
    # solve ODE
    z = solve_ivp(model,[self.t[self.cont], self.t[self.cont+1]],[self.u[0],self.u1[0]],t_eval=[self.t[self.cont+1]],method='LSODA',args=(action_vec,))
    
    #--------------
    self.u, self.u1 = z.y[0], z.y[1]
    
    # Update iteration cont
    self.cont += 1

    # Reward and Penalty function
    if abs(np.max(self.u)) < self.u_lim:
        reward = (abs(self.u)**2 + abs(self.u1)**2)
    else:
        reward = 1e5
        
    # Exit condition
    if abs(np.max(self.u)) > self.u_lim or self.cont == len(self.t)-1:
        done = True
    else:
        done = False  

    # Observation collection       
    observation  = np.array([self.u[0],self.u1[0]])
    
    info = {} # Optional
    return observation, reward, done, info

  def reset(self):
    """
    This funciton reset the simulation to its starting point
    Important: the observation must be a numpy array
    :return: (np.array)
    """
    # Define the displacement vectors at different time steps
    self.u = np.array([4])
    self.u1  = np.array([0])
    
    # Initialize the conunter
    self.cont = 0
    self.action_replay.clear()

    # Observation collection
    observation  = np.array([self.u[0],self.u1[0]])

    return observation

  def render(self, mode='human'):
      pass
      
  def close (self):
      pass

