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
        self.gamma = 0.9
        self.epsilon = 0.5
        self.Q_table = defaultdict(dict)
        self.cumulative_reward = 0
        self.n_success = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.cumulative_reward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.update_state(self.next_waypoint,
                                       inputs['light'],
                                       inputs['oncoming'],
                                       inputs['left'])

        # TODO: Select action according to your policy
        if random.random() < self.epsilon:  # explore
            action = random.choice((None, 'forward', 'left', 'right'))
        else:  # exploit
            if self.Q_table.has_key(self.state):  # encountered this state previously
                # select an action which maximize the Q value
                actions = self.Q_table[self.state]
                try:
                    action = max(actions, key=actions.get)
                except ValueError:  # when there is no action corresponding to the state
                    action = random.choice((None, 'forward', 'left', 'right'))
            else:
                # never encountered this state before
                action = random.choice((None, 'forward', 'left', 'right'))
                # initialize the Q value of the state and the action with a certain value
                # self.Q_table[self.state][action] = 0.5

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.n_success += 1 if reward == 12 else 0
        self.cumulative_reward += reward

        # next state
        next_waypoint_prime = self.planner.next_waypoint()
        inputs_prime = self.env.sense(self)
        state_prime = self.update_state(next_waypoint_prime,
                                        inputs_prime['light'],
                                        inputs_prime['oncoming'],
                                        inputs_prime['left'])

        # TODO: Learn policy based on state, action, reward
        action_prime_values = self.Q_table[state_prime].values() if self.Q_table[state_prime].values() else [0]
        self.Q_table[self.state][action] = (1 - self.alpha) * self.Q_table[self.state].get(action, 0) \
                                       + self.alpha * (reward + self.gamma * max(action_prime_values))

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, cumulative_reward = {}".format(deadline, inputs, action, reward, self.cumulative_reward)  # [debug]
        # print self.Q_table [debug]
        print self.state
        print state_prime

    def update_state(self, *args):
        state = ''
        for arg in args:
            state += arg if arg else 'None'
        return state


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.3, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
