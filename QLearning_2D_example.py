'''
This is an extension example of applying tabular Q-Learning algorithm in 2D environment.
You could check my Q-learning first try 1D environment example in QLearning_example.py
References:
1.莫烦python https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-general-rl/
'''

import numpy as np
import pandas as pd
import time
import os

# Environment: a two dimensional world
# e.g. 6*6 env
# -o----
# ------
# ------
# ------
# ----T-
# ------
# o: location of agent
# T: location of treasure

# State of agent: location of agent in the world (xa,ya)
# State/observation of the total environment: location of agent and treasure ((xa,ya),(xt,yt))
# Action of agent: up/down/left/right

np.random.seed(10000)  # reproducible

N_STATUS = 6  # height/width of the world
L_TREASURE = [np.random.randint(0, N_STATUS),np.random.randint(0, N_STATUS)]
L_AGENT = [0, 0]
ACTIONS = ['up','down','left', 'right']

EPISILON = 0.9  # greedy
LEARNING_RATE = 0.1  # learning rate
LAMDA = 0.9  # discount factor
MAX_EPISODES = 30  # max episodes
FRESH_TIME = 0.5  # move Interval

MAX_STEP = 2*N_STATUS*N_STATUS

def init_env():
    x_AGENT= np.random.randint(0, N_STATUS)
    y_AGENT=np.random.randint(0,N_STATUS)
    c_AGENT=[x_AGENT, y_AGENT]
    if c_AGENT[0]==L_TREASURE and c_AGENT[1]== L_TREASURE[1]:
        c_AGENT=init_env()
    else:
        L_AGENT=c_AGENT
    return L_AGENT

# For agent, the location of treasure is unseen, so the s in Q(s,a）is just the location of agent itself
# Define Q-table
# q_table:
"""
    up     down   left   right
0   0.0    0.0    0.0    0.0
1   0.0    0.0    0.0    0.0
2   0.0    0.0    0.0    0.0
...
34  0.0    0.0    0.0    0.0
35  0.0    0.0    0.0    0.0
"""
def build_q_table(n_status, actions):
    # initialize all items in q-table 0
    table=pd.DataFrame(np.zeros((n_status*n_status,len(actions))), columns=actions,)
    return table

# epsilon greedy: At the beginning, random explorations are more useful than definite strategy
# so it should be not too greedy at first, and the greedy degree should increase as processing.
# However, in this realization, we just set a fixed greedy degree as 0.9 to simplify.

# choose action at state s, according to q-table
def choose_action(state, q_table):
    state_action_values= q_table.iloc[state,:]
    if np.random.uniform()>EPISILON or state_action_values.all()==0:
        # not greedy
        action_name=np.random.choice(ACTIONS)
    else:
        # greedy
        action_name=state_action_values.idxmax()

    return action_name

# How agent will interactive with environment
# what will happen when agent take action A under state S
# the next state S_ they will enter, and the reward R agent will get
# define that the agent will only get a reward when it find the treasure, and no reward otherwise
def step1(S,A):
    x=S[0]
    y=S[1]
    if A == 'up':
        if x == L_TREASURE[0] and y== L_TREASURE[1]+1:
            S_='terminal'
            R=1
        elif y == 0:
            # hit the wall
            S_=S
            R=0
        else:
            S_=[S[0],S[1]-1]
            R=0
    elif A== 'down':
        if x == L_TREASURE[0] and y== L_TREASURE[1]-1:
            S_='terminal'
            R=1
        elif y==N_STATUS-1:
            S_=S
            R = 0
        else:
            S_=[S[0],S[1]+1]
            R=0
    elif A=='left':
        if x == L_TREASURE[0]+1 and y== L_TREASURE[1]:
            S_='terminal'
            R=1
        elif x == 0:
            # hit the wall
            S_=S
            R=0
        else:
            S_=[S[0]-1,S[1]]
            R=0
    else:
        if x == L_TREASURE[0]-1 and y== L_TREASURE[1]:
            S_='terminal'
            R=1
        elif x==N_STATUS-1:
            S_=S
            R = 0
        else:
            S_=[S[0]+1,S[1]]
            R=0
    return S_,R

def step2(S,A):
    x=S[0]
    y=S[1]
    vector_tresure=np.array(L_TREASURE)
    o_dis = np.linalg.norm(np.array(S) - vector_tresure, ord=1)
    if A == 'up':
        if x == L_TREASURE[0] and y== L_TREASURE[1]+1:
            S_='terminal'
            R=1
        elif y == 0:
            # hit the wall
            S_=S
            R=0
        else:
            S_=[S[0],S[1]-1]
            n_dis=np.linalg.norm(np.array(S_)-vector_tresure,ord=1)
            R= float(o_dis-n_dis)/2
    elif A== 'down':
        if x == L_TREASURE[0] and y== L_TREASURE[1]-1:
            S_='terminal'
            R=1
        elif y==N_STATUS-1:
            S_=S
            R = 0
        else:
            S_=[S[0],S[1]+1]
            n_dis = np.linalg.norm(np.array(S_) - vector_tresure, ord=1)
            R = float(o_dis - n_dis) / 2
    elif A=='left':
        if x == L_TREASURE[0]+1 and y== L_TREASURE[1]:
            S_='terminal'
            R=1
        elif x == 0:
            # hit the wall
            S_=S
            R=0
        else:
            S_=[S[0]-1,S[1]]
            n_dis = np.linalg.norm(np.array(S_) - vector_tresure, ord=1)
            R = float(o_dis - n_dis) / 2
    else:
        if x == L_TREASURE[0]-1 and y== L_TREASURE[1]:
            S_='terminal'
            R=1
        elif x==N_STATUS-1:
            S_=S
            R = 0
        else:
            S_=[S[0]+1,S[1]]
            n_dis = np.linalg.norm(np.array(S_) - vector_tresure, ord=1)
            R = float(o_dis - n_dis) / 2
    return S_,R

def update_env(S,episode,step_counter, is_terminal):
    # This is how environment be updated after every step
    # TODO: Need Modify
    lt=L_TREASURE[0]*N_STATUS+L_TREASURE[1]
    sq=N_STATUS*N_STATUS
    env_list = ['-'] * lt + ['T'] +  ['-'] * (sq-1-lt)
    if S == 'terminal':
        if is_terminal:
            interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
            print('\r{}'.format(interaction), end='')
            time.sleep(2)
            print('\r                                ', end='')
        else:
            interaction = 'Episode %s: total_steps = %s, the agent can\' t find treasure' % (episode + 1, step_counter)
            print('\r{}'.format(interaction), end='')
            time.sleep(2)
            print('\r                                ', end='')
    else:
        ls=S[0]*N_STATUS+S[1]
        env_list[ls] = 'o'
        interaction=[]
        for i in range(N_STATUS):
            interi = ''.join(env_list[i*N_STATUS:(i+1)*N_STATUS])
            interaction.append(interi)
        interaction = '\n'.join(interaction)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    table=build_q_table(N_STATUS,ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter=0
        S=init_env()
        is_terminal= False

        update_env(S,episode=episode,step_counter=step_counter,is_terminal=is_terminal)
        while not is_terminal:
            ss=S[0]*N_STATUS+S[1]
            A=choose_action(ss,table)
            S_,R=step2(S,A)
            q_predict=table.loc[ss,A] # estimated Q(s,a) value

            # It's similar to the recursive in dynamic programming problem（remember the recursive table?)
            ss_=S_[0]*N_STATUS+S_[1]
            if S_!="terminal":
                q_target = R+LAMDA*table.iloc[ss_,:].max()
            else:
                q_target = R
                is_terminal = True

            # update q_table
            table.loc[ss,A] += LEARNING_RATE*(q_target-q_predict)
            S = S_
            step_counter += 1
            if not is_terminal and step_counter == MAX_STEP:
                S = "terminal"
                update_env(S, episode, step_counter, is_terminal)
                is_terminal = True
            else:
                update_env(S,episode,step_counter,is_terminal)
    return table

if __name__=="__main__":
    q_table=rl()
    print('\r\nQ-table:\n')
    print(q_table)
