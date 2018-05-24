'''
This is a simple example of applying tabular Q-Learning algorithm
References:
1.莫烦python https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-general-rl/
'''

import numpy as np
import pandas as pd
import time

# Environment: a one dimensional world
# e.g. -o----T
# o: location of agent
# T: location of treasure

# State of agent: location of agent in the world
# State/observation of the total environment: location of agent and treasure
# Action of agent: left/right

np.random.seed(10000)  # reproducible

N_STATUS = 10  # width of the world
L_TREASURE = np.random.randint(0, N_STATUS)
L_AGENT = 0
ACTIONS = ['left', 'right']

EPISILON = 0.9  # greedy
LEARNING_RATE = 0.1  # learning rate
LAMDA = 0.9  # discount factor
MAX_EPISODES = 30  # max episodes
FRESH_TIME = 0.3  # move Interval

MAX_STEP = 2*N_STATUS

def init_env():
    # print(L_TREASURE)
    l_AGENT= np.random.randint(0, L_TREASURE)
    r_AGENT=np.random.randint(L_TREASURE+1,N_STATUS)
    L_AGENT=np.random.choice([l_AGENT, r_AGENT])
    return L_AGENT
    # L_TREASURE=l_treasure

# For agent, the location of treasure is unseen, so the s in Q(s,a）is just the location of agent itself
# Define Q-table
# q_table:
"""
   left  right
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0
"""
def build_q_table(n_status, actions):
    # initialize all items in q-table 0
    table=pd.DataFrame(np.zeros((n_status,len(actions))), columns=actions,)
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
def step(S,A):
    if A == 'left':
        if S == L_TREASURE+1:
            S_='terminal'
            R=1
        elif S == 0:
            # hit the wall
            S_=S
            R=0
        else:
            S_=S-1
            R=0
    else:
        if S==L_TREASURE-1:
            S_='terminal'
            R=1
        elif S==N_STATUS-1:
            S_=S
            R = 0
        else:
            S_=S+1
            R=0
    return S_,R

def update_env(S,episode,step_counter, is_terminal):
    # This is how environment be updated after every step
    env_list = ['-'] * (L_TREASURE) + ['T'] +  ['-'] * (N_STATUS-1-L_TREASURE) # '-----T----' our environment
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
        env_list[S] = 'o'
        interaction = ''.join(env_list)
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
            A=choose_action(S,table)
            S_,R=step(S,A)
            q_predict=table.loc[S,A] # estimated Q(s,a) value

            # It's similar to the recursive in dynamic programming problem（remember the recursive table?)
            if S_!="terminal":
                q_target = R+LAMDA*table.iloc[S_,:].max()
            else:
                q_target = R
                is_terminal = True

            # update q_table
            table.loc[S,A] += LEARNING_RATE*(q_target-q_predict)
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







