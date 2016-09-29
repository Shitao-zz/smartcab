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
        self.trial = 0
	self.total_reward = 0
	self.negative_reward = 0
        self.destination_reached = False
	self.trial_stats = pd.DataFrame(columns=['total_reward','negative_reward','trial_length'])

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.trial = 0
	self.total_reward = 0
	self.negative_reward = 0
	self.destination_reached = False
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Follow traffic rules
        action_okay = True
        if self.next_waypoint == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif self.next_waypoint == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        elif self.next_waypoint == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False
        action = None
        if action_okay:
            action = self.next_waypoint
        # TODO: Select action according to your policy
        #action = random.choice([None,'forward','left','right'])
        
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
        # TODO: Learn policy based on state, action, reward

       # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def save_trial_stats(self):

        trial_df = pd.DataFrame([[self.total_reward, self.negative_reward, self.trial]],columns = ['total_reward','negative_reward','trial_length']) 
        if self.trial_stats.empty:
            self.trial_stats = trial_df
        else:
            self.trial_stats = self.trial_stats.append(trial_df, ignore_index = True)
        if self.trial_stats.shape[0] == 100:
            self.trial_stats.to_csv('planner_only.csv',index=False)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
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
