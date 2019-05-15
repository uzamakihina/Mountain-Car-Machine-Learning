
from rl_glue import BaseAgent
import numpy as np
from tile3 import IHT, tiles
import random


class Agent(BaseAgent):

    def __init__(self):
        # create needed variables
        self.w = None
        self.pos_scale = 8/1.7
        self.vel_scale = 8/0.14
        self.iht = None
        self.z = None
        
        self.last_list = None
        self.squiggle = None
        self.epis = None
        # using alpha of 0.4/8
        self.alpha = 0.1/8
    
    
    def agent_init(self):
        
        # set needed variables to values from preconditions
        self.w = np.array([0]*2048)
        
        # random initial values
        for i in range(2048):
            self.w[i] = random.uniform(-0.001,0)
        
        # hash table
        self.iht = IHT(2048)
        self.epis = 0
    
        

    def agent_start(self, state):
        
        # reset z
        self.z = np.array([0]*2048)
        
        # call choose which gives a state and returns the optimal action A and the indexs  of the weight functions in a list B
        
        A,B = self.choose(state)
       
       # save last tile coder list return
        self.last_list = B
        
    
        return A

    def agent_step(self, reward, state):
        
        
        # call choose which gives a state and returns the optimal action A and the indexs  of the weight functions in a list B
        A,B = self.choose(state)
        
        # update the weight vector
        
        squiggle = reward
        for i in self.last_list:
            squiggle = squiggle - self.w[i]
            self.z[i] = 1


        for i in B:
            squiggle = squiggle + self.w[i]
        
        self.w = self.w + squiggle*self.z*self.alpha
        self.z = 0.9*self.z
        
        # save the previos tile coder return so we dont have to call tile coder as often
        self.last_list = B
            
        # return action
        return A
    
    
    def agent_end(self,reward):
        
        # end weight vector update
        squiggle = reward
        for i in self.last_list:
            squiggle = squiggle - self.w[i]
            self.z[i] = 1
        
        self.w = self.w + self.alpha*squiggle*self.z
        
        
        
        self.epis += 1
        
        print("episode ", self.epis , " done")
    
    
    def agent_message(self, message):
        # return a list of max q values per state action in a equal 50 seperate sapce per dimention
        if message == 1:
            state = []
            x = -1.2        # position
            y = -0.07       # velocity
            for i in range(50):
                y = y + 0.0028
                x = -1.2
                for e in range(50):
                    A,B = self.choose([x,y])
                    
                    temp = 0
                    for num in B:
                        temp = temp + self.w[num]
                    state.append([x,y,temp])
                    x = x + 0.034
    
        return state
    
    
    # Function used to choose between three actions and chooses the optimal action and returns the corresponding weigt vector indexes

    def choose(self, state):

# find the value of each action
        temp1 = tiles( self.iht, 8, [state[0]*self.pos_scale , state[1]*self.vel_scale], [0] )
        a = 0
        b = 0
        c = 0
        d = [a,b,c]
        
        
        for i in temp1:
            a = a + self.w[i]


        temp2 = tiles( self.iht, 8, [state[0]*self.pos_scale , state[1]*self.vel_scale], [1] )
        for i in temp2:
            b = b + self.w[i]



        temp3 = tiles( self.iht, 8, [state[0]*self.pos_scale , state[1]*self.vel_scale], [2] )
        for i in temp3:
            c = c + self.w[i]

        all = [temp1,temp2,temp3]

#       Tie breaking if the actions have same values and return a random one from the ties
        if (a == b) or (b ==c) or (a ==c):
            # if their all equal
            if (a ==b) and (b ==c):
                r = np.random.randint(0,3)
        
                return r , np.array(all[r])
            
            # if a=b and c is less than both
            if (a == b) and c < a:
                r = np.random.randint(0,2)
                
                return r, np.array(all[r])
            # if c is the largest
            elif (a == b) and a < c:
                return 2, np.array(all[2])
            
    
            # if b=c and a < c
            if (b==c) and a < c:
                r= np.random.randint(1,3)
            
                return r, np.array(all[r])
            # if a is the largest
            elif (b==c) and c < a:
                return 0, np.array(all[0])
            
            #  if a == c

            if b > a:
                return 1, np.array(all[1])
            else:
                temp = [0,2]
                r = temp[np.random.randint(0,2)]
                return r , np.array(all[r])
            
            

        # return max action and its weight vector index
        temp = [a,b,c]
        max = a
        idx = 0
        for i in range(1,3):
            if temp[i] > max:
                max = temp[i]
                idx = i
        
        return idx, np.array(all[idx])
        













   






































































































