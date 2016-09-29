import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
import numpy as np
random.seed(0)

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # Trial statistics
	self.trial = 0
	self.total_reward = 0
	self.negative_reward = 0
	self.trial_stats = pd.DataFrame(columns=['total_reward','negative_reward','trial'])
	self.destination_reached = False
        # Q learning variables
	self.alpha = 0.5
	self.gamma = 0.5
	self.epsilon = 0.1
        self.actions = self.env.valid_actions
	self.states = [(light, oncoming,left,next_waypoint)
	    for light in ('red','green')
            for oncoming in ('forward','left','right',None)
            for left in ('forward','left','right',None)
	    for next_waypoint in ('forward','left','right')
	           ]
        self.Q = { s:[0]*len(self.actions) for s in self.states }

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.trial = 0
	self.total_reward = 0
	self.negative_reward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        #self.state = (inputs['light'], inputs['oncoming'],inputs['left'])
	# TODO: Select action according to your policy
	Qmax = self.Q[self.state].index(max(self.Q[self.state]))
        
	# Q learn or new path?
	if random.uniform(0,1) < self.epsilon:
	    action = random.choice(self.env.valid_actions)
	else:
	    action = self.actions[Qmax]
        # Execute action and get reward
	reward = self.env.act(self, action)

	# Trial statistics
	self.total_reward += reward
	if reward < 0:
	    self.negative_reward += reward
	self.trial += 1
        self.destination_reached = reward > 2
	if self.destination_reached or deadline == 0: 
	    self.save_trial_stats() 
 
	# Learn policy based on state, action and reward
        # Q learn equation: Q(s,a) = (1-alpha)*Q(s,a) + alpha*(reward + gamma*argmaxQ(s',a'))
	# get Q(s',a')
        new_input = self.env.sense(self) 
        new_next_waypoint = self.planner.next_waypoint()
	new_state = (new_input['light'],new_input['oncoming'],new_input['left'],new_next_waypoint)
	
        # Update the Q matrix
	self.Q[self.state][self.actions.index(action)] = (1-self.alpha)*self.Q[self.state][self.actions.index(action)] + \
	    (self.alpha*(reward + self.gamma*max(self.Q[new_state])))
	
       
    def save_trial_stats(self):
        print "$$$$$$$$$"
        trial_df = pd.DataFrame([[self.total_reward, self.negative_reward, self.trial]],columns = ['total_reward','negative_reward','trial']) 
        if self.trial_stats.empty:
            self.trial_stats = trial_df
        else:
            self.trial_stats = self.trial_stats.append(trial_df, ignore_index = True)
        if self.trial_stats.shape[0] == 100:
            print "###########"
            self.trial_stats.to_csv('Q_learning.csv')


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    print e.wins
    print e.crashes

if __name__ == '__main__':
    run()
