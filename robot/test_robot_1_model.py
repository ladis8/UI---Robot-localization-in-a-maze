"""
Example scripts for Robot in a maze HMM

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from hmm_inference import *
from robot_1_model import *
from utils import normalized
import copy

from time import sleep

# direction_probabilities = {
#    NORTH: 0.25,
#    EAST: 0.25,
#    SOUTH: 0.25,
#    WEST: 0.25
# }

NORTH = (-1, 0)
EAST = (0, 1)
SOUTH = (1, 0)
WEST = (0, -1)
MY_DIRS = (NORTH, EAST, SOUTH, WEST) # none missing

N, E, S, W = 0, 1, 2, 3
ALL_ORIS = (N, E, S, W)

def init_maze():
   """Create and initialize robot instance for subsequent test"""
   m = Maze('mazes/rect_6x10_obstacles.map')
   # robot = Robot(MY_DIRS, direction_probabilities)
   robot = Robot()
   robot.maze = m
   robot.position = (2, 4, 0)
   robot.set_active_sensors()
   print_robot(robot.position, robot.maze.map)
   print('Robot at ', robot.position)
   print('=' * 30)
   return robot


def test_pt():
   """Try to compute transition probabilities for certain position"""
   robot = init_maze()
   robot.position = (2, 10, 2)
   robot.set_active_sensors()
   print('Robot at', robot.position)
   print_robot(robot.position, robot.maze.map)
   for fp in robot.maze.get_free_positions():
      for ori in ALL_ORIS:
         pos = (fp[0], fp[1], ori)
         p = robot.pt(robot.position, pos)
         if p > 0:
            print('Prob of transition to', pos, 'is', p)


def test_pe():
   """Try to compute the observation probabilities for certain position"""
   robot = init_maze()
   robot.position = (1, 6, 3)
   robot.set_active_sensors()
   print('Robot at', robot.position)
   print_robot(robot.position, robot.maze.map)
   for obs in robot.get_observations():
      p = robot.pe(robot.position, obs)
      if p > 0:
         print('Prob obs', obs, 'is', p)


def test_simulate():
   """Try to generate some data from the robot domain"""
   robot = init_maze()
   print('Generating data...')
   states, observations = robot.simulate(n_steps=5)
   for i, (s, o) in enumerate(zip(states, observations)):
      print('Step:', i + 1, '| State:', s, '| Observation:', o)
      print_robot(s, robot.maze.map)
      sleep(0.5)


def test_filtering():
   """Try to run filtering for robot domain"""
   robot = init_maze()
   #    states, obs = robot.simulate(init_state=(1,1), n_steps=3)
   states, obs = robot.simulate(n_steps=10)
   print('Running filtering...')
   initial_belief = normalized({pos: 1 for pos in robot.get_states()})
   beliefs = forward(initial_belief, obs, robot)
   for state, belief in zip(states, beliefs):
      print('Real state:', state)
      print_robot(state, robot.maze.map)
      print('Sorted beliefs:')
      for k, v in sorted(belief.items(), key=lambda x: x[1], reverse=True):
         if v > 0:
            print(k, ':', v)


def test_smoothing():
   """Try to run smoothing for robot domain"""
   robot = init_maze()
   states, obs = robot.simulate(n_steps=10)
   print('Running smoothing...')
   initial_belief = normalized({pos: 1 for pos in robot.get_states()})
   beliefs = forwardbackward(initial_belief, obs, robot)
   for state, belief in zip(states, beliefs):
      print('Real state:', state)
      print('Sorted beliefs:')
      for k, v in sorted(belief.items(), key=lambda x: x[1], reverse=True):
         if v > 0:
            print(k, ':', v)


def test_viterbi():
   """Try to run Viterbi alg. for robot domain"""
   robot = init_maze()
   states, obs = robot.simulate(n_steps=30)
   print('Running Viterbi...')
   initial_belief = normalized({pos: 1 for pos in robot.get_states()})
   ml_states, max_msgs = viterbi(initial_belief, obs, robot)
   for real, est in zip(states, ml_states):
      print('Real pos:', real, '| ML Estimate:', est)


def print_robot(pos, maze):
   char_ori = ['^', '>', 'v', '<']
   ori = pos[2]
   pos = (pos[0], pos[1])
   for i in range(len(maze)):
      for ii in range(len(maze[0])):
         if pos == (i, ii):
            print(char_ori[ori], end='')
            continue
         print(maze[i][ii], end='')
      print('\n', end='')


if __name__ == '__main__':
   print('Uncomment some of the tests in the main section')
   # test_pt()
   # test_pe()
   # test_simulate()
   # test_filtering()
   test_smoothing()
   test_viterbi()
