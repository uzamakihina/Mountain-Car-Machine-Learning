#!/usr/bin/env python

import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
from rl_glue import RLGlue
from environment import Environment
from mpl_toolkits.mplot3d import axes3d, Axes3D
from plot import graph
import os

def question_1():
    # Specify hyper-parameters

    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    num_episodes = 200
    num_runs = 50
    max_eps_steps = 100000

    steps = np.zeros([num_runs, num_episodes])

    for r in range(num_runs):
        print("question 1 run number : ", r)
        rlglue.rl_init()
        for e in range(num_episodes):
            rlglue.rl_episode(max_eps_steps)
            steps[r, e] = rlglue.num_ep_steps()
            # print(steps[r, e])
    np.save('steps', steps)
    del agent, environment,rlglue

# used for plotting 3d graph
def question_3():

    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)
    
    num_episodes = 1000
    num_runs = 1
    max_eps_steps = 100000
    
    steps = np.zeros([num_runs, num_episodes])
    # only 1 run
    for r in range(num_runs):
        print("1000 episode run : ", r)
        rlglue.rl_init()
        for e in range(num_episodes):
            rlglue.rl_episode(max_eps_steps)
            steps[r, e] = rlglue.num_ep_steps()
        # get the list of value functions [X,Y,Z] represents position, velocity, state-value
        Return = rlglue.rl_agent_message(1)
    return Return




if __name__ == "__main__":
    # first graph method
    question_1()
    # generate 2d graph using provided plot file
    graph()
    
    #initialize a list of all state values
    Z = [0]*2500
    X = [0]*2500
    Y = [0]*2500
    #generate data for the 1000run 3d graph
    a = question_3()
    #seperate data
    for i in range(2500):
        Z[i] = -a[i][2]
        X[i] = a[i][0]
        Y[i] = a[i][1]
    
    # reshapte data so it fits in a 3d form
    Z = np.array(Z).reshape((50,50))
    X = np.array(X).reshape((50,50))
    Y = np.array(Y).reshape((50,50))
    fig = plt.figure()
    ax = Axes3D(fig)
    # label the axis
    plt.xlabel("position")
    plt.ylabel("velocity")
    ax.set_zlabel(" - q_value")
    # save figure
    surf = ax.plot_surface(X, Y, Z,)
    plt.savefig(" generated negative value function 3d graph" )
    os.remove("steps.npy")











