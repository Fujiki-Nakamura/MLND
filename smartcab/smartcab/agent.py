from collections import defaultdict
import random

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha = 0.5
        self.gamma = 0.5
        self.epsilon = 0.5
        self.Q_hat = defaultdict(dict)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = inputs['light']
        self.state += inputs['oncoming'] if inputs['oncoming'] else 'None'
        self.state += inputs['right'] if inputs['right'] else 'None'
        self.state += inputs['left'] if inputs['left'] else 'None'

        # TODO: Select action according to your policy
        if random.random() < self.epsilon:  # exploit
            if self.Q_hat.has_key(self.state):  # encountered this state previously
                # select an action which maximize the Q value
                actions = self.Q_hat[self.state]
                try:
                    action = max(actions, key=actions.get)
                except ValueError:  # when there is no action corresponding to the state
                    action = random.choice((None, 'forward', 'left', 'right'))
            else:
                # never encountered this state
                action = random.choice((None, 'forward', 'left', 'right'))
                # initialize the Q value of the state and the action with a certain value
                self.Q_hat[self.state][action] = 0.5
        else:  # explore
            action = random.choice((None, 'forward', 'left', 'right'))

        # Execute action and get reward
        reward = self.env.act(self, action)

        # next state
        inputs_prime = self.env.sense(self)
        state_prime = inputs_prime['light']
        state_prime = state_prime + inputs_prime['oncoming'] if inputs_prime['oncoming'] else ''
        state_prime = state_prime + inputs_prime['right'] if inputs_prime['right'] else ''
        state_prime = state_prime + inputs_prime['left'] if inputs_prime['left'] else ''

        # TODO: Learn policy based on state, action, reward
        action_prime_values = self.Q_hat[state_prime].values() if self.Q_hat[state_prime].values() else [0]
        self.Q_hat[self.state][action] = (1 - self.alpha) * self.Q_hat[self.state].get(action, 0) \
                                       + self.alpha * (reward + self.gamma * max(action_prime_values))

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print self.Q_hat

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
