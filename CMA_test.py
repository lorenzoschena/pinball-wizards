# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 17:59:58 2021

@author: fabio
"""
import cma
import numpy as np
import matplotlib.pyplot as plt
import time

'''
This script find an open-loop solution to the control of an ODE using a set of Radial Basis Function in the time domain parametrized in terms of mean and variance. The optimal parameters are found using a Covariance Matrix Adaptation library
'''
# Importing the training env
from env_RL import system as CustomEnv

# Initializing the environment object
ambiente = CustomEnv()
    
# Radial Basis Function parametrization
centers = [1,2,5,7,7.5,8] # mean values
epsilons = [2,3,3,3,2,3] # standard deviations
    
# Radial basis function 
def RBF(distance,epsilon):
    '''
    This function compute the RBF of the distance between a point t in the time space with standard deviation epsilon
    '''
    return np.exp(-(epsilon*distance)**2)
    
t = 10

# time vector
t_list = np.linspace(0,10,1000)

# approximate optimal open-loop control function
y_approx = np.zeros(len(t_list))
    
for jj in range(len(t_list)):
    for i in range(len(centers)):
        y_approx[jj] += 2*RBF(np.linalg.norm(t_list[jj]-centers[i]),epsilons[i])

# Plotting the different RBFs
for ii in range(len(centers)):
    plt.plot(t_list,RBF(np.abs(t_list-centers[ii]),epsilons[ii]))
    
plt.plot(t_list,y_approx,'k-')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(f"pics.png", dpi = 400,bbox_inches='tight') 
plt.show()
    
# Bounds for the search space
centers_min = 0
centers_max = 30
weight_min = -20
weight_max = 20
epsilons_min = 0.01
epsilons_max = 10
    
action_max = 100
action_min = -100
    
# number of RBFs
N = 10
img_cont = 0
for iii in range(40):
    
    observation, reward, _, _ = ambiente.step(4)
    
    # print(observation)

def FUNCTION(INPUT):
    
    # Scaling the inputs from [-1 1] to [0 1]
    INPUT = (INPUT+1)/(2)
    
    weights = INPUT[:N]*(weight_max - weight_min) + weight_min
    centers = INPUT[N:2*N]*(centers_max - centers_min) + centers_min
    epsilons = INPUT[2*N:3*N]*(epsilons_max - epsilons_min) + epsilons_min
    
    done = False
    total_reward = 0
    contatore = -1
    t_list = np.linspace(0,30,10000)
    ambiente.reset()
    s1 =  []
    s2 = []
    while not done:
        contatore += 1
        action = 0
        for i in range(len(centers)):
            action += weights[i]*RBF(np.linalg.norm(t_list[contatore]-centers[i]),epsilons[i])
            
        # action = (action_max - action)/(action_max - action_min)
        observation, reward, done, _ = ambiente.step(np.array(action))
        s1.append(observation[0])
        s2.append(observation[1])
        total_reward += reward
        
    # Computing the approximate optimal open-loop control function
    y_approx = np.zeros(len(t_list))
        
    for jj in range(len(t_list)):
        for i in range(len(centers)):
            y_approx[jj] += weights[i]*RBF(np.linalg.norm(t_list[jj]-centers[i]),epsilons[i])
    global img_cont
    
    for ii in range(len(centers)):
        plt.plot(t_list,RBF(np.abs(t_list-centers[ii]),epsilons[ii]),'-',linewidth=0.8)
        
    plt.plot(t_list,y_approx,'k-')
    plt.grid()
    plt.title("Merlin")
    plt.savefig(f"pics/{img_cont}.png", dpi = 400,bbox_inches='tight') 
    plt.show()
    plt.close()
    # time.sleep(0.3)
    
    # plt.plot(s1,'k-',label='s1')
    # plt.plot(s2,'r--',label='s2')
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    # # Plotting the evolution of the system
    # plt.figure()
    # plt.plot(s1,label='displacement')
    # plt.plot(s2,label='velocity')
    # plt.title('Evolution of the dynamical system')
    # plt.legend()
    # plt.show()
    # plt.close()
    
    img_cont += 1
    
    print(f'Total Reward: {total_reward}')
    return(total_reward)
    

# Defining a CMA object with the number of parameters and the initial variance of the covariance matrix
es = cma.CMAEvolutionStrategy(3*N * [0], 0.5)

while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [FUNCTION(x) for x in solutions])
    es.logger.add()  # write data to disc to be plotted
    es.disp()
    
    
    # plt.imshow(ZZZ, extent=[0, 5, 0, 5], origin='lower',cmap='RdGy', alpha=0.5)
    # plt.colorbar();
    
    # for iii in range(len(solutions)):
    
    #     plt.plot(solutions[iii][0], solutions[iii][1], 'r.')
    # plt.grid()
    # plt.show()
    # time.sleep(0.3)
    
es.result_pretty()

cma.plot()  # shortcut for es.logger.plot()