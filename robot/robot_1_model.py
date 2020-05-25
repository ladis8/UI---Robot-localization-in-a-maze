"""
Implementation of Robot-in-a-maze domain as Hidden Markov Model.

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
Part of semestral project 2: Robot localization using HMM
"""

import random
from collections import Counter
from itertools import product

from utils import weighted_random_choice, normalized
from probability_vector import ProbabilityVector, ProbabilityMatrix

from hmm import HMM
import random

NORTH = (-1, 0)
EAST = (0, 1)
SOUTH = (1, 0)
WEST = (0, -1)
ALL_DIRS = (NORTH, EAST, SOUTH, WEST)

# DEFAULT_MOVE_PROBS = {
#    NORTH: 0.25,
#    EAST: 0.25,
#    SOUTH: 0.25,
#    WEST: 0.25
# }

FWD, TURN_RIGHT, TURN_LEFT = 0, 1, 2

FWD_MOVE_PROBS = {
   FWD: 2 / 3,
   TURN_RIGHT: 1 / 6,
   TURN_LEFT: 1 / 6
}

# FWD_MOVE_PROBS = {
#    FWD: 1/3,
#    TURN_RIGHT: 1/3,
#    TURN_LEFT: 1/3
# }

N, E, S, W = 0, 1, 2, 3
ALL_ORIS = (N, E, S, W)

SENSORS_SWITCH = {
   0: [0, 1, 3],
   1: [0, 1, 2],
   2: [1, 2, 3],
   3: [0, 2, 3]
}


def add(s, dir):
   """Add direction to a state"""
   return tuple(a + b for a, b in zip(s, dir))


# def direction(s1, s2):
#    """Return a direction vector from s1 to s2"""
#    return tuple(b - a for a, b in zip(s1, s2))


# def manhattan(s1, s2):
#    return sum(abs(a) for a in direction(s1, s2))


class Maze:
   """Representation of a maze"""

   WALL_CHAR = '#'
   FREE_CHAR = ' '

   def __init__(self, map):
      self.load_map(map)
      self._free_positions = []

   def load_map(self, map_fname):
      """Load map from a text file"""
      self.map = []
      with open(map_fname, 'rt', encoding='utf-8') as f:
         for line in f:
            line = line.strip()
            if not line: continue
            self.map.append(line)
      self.map = tuple(self.map)
      self.height = len(self.map)
      self.width = len(self.map[0])
      self.min_pos = (1, 1)
      self.max_pos = (len(self.map) - 2, len(self.map[0]) - 2)

   def __str__(self):
      """Return a string representation of the maze"""
      return '\n'.join(self.map)

   def is_free(self, pos):
      """Check whether a position is free

      :param pos: position as (row, column)
      :return: True or False
      """
      return self.map[pos[0]][pos[1]] == self.FREE_CHAR

   def is_wall(self, pos):
      """Check whether a position contains a wall

      :param pos: position as (row, column)
      :return: True or False
      """
      return self.map[pos[0]][pos[1]] == self.WALL_CHAR

   def get_free_positions(self, search=False):
      """Return a list of all free positions in the maze

      It returns a cached list, if the free positions were already precomputed.
      """
      # If free positions already found and shouldn't find them anew
      if self._free_positions and not search:
         return self._free_positions
      fp = []
      for r in range(1, self.height - 1):
         for c in range(1, self.width - 1):
            pos = (r, c)
            if self.is_free(pos):
               fp.append(pos)
      self._free_positions = fp
      return fp

   def get_wall_positions(self):
      """Return a list of all maze positions containing walls"""
      wp = []
      for r in range(0, self.height):
         for c in range(0, self.width):
            pos = (r, c)
            if self.is_wall(pos):
               wp.append(pos)
      return wp

   def get_dist_to_wall(self, pos, dir):
      """Return the distance to the nearest wall.

      Distance is the number of steps one can make from
      the given position to the closest wall in the given direction.

      :param pos: (row, column) tuple
      :param dir: One of the following:
                  (-1, 0) ... north
                  (0, 1) ... east
                  (1, 0) ... south
                  (0, -1) ... west
      :return: int, distance to wall
      """
      if pos not in self.get_free_positions():
         raise ValueError('The specified robot position is not allowable. Did you use (row, column)?')
      d = 0
      while True:
         pos = add(pos, dir)
         if not self.is_free(pos):
            return d
         d += 1


class NearFarSensor:
   """Crude sensor measuring direction"""

   VALUES = ['n', 'f']
   DIST_OF_SURE_FAR = 4  # For this distance and larger, the sensor will surely return 'f'

   def __init__(self, robot, direction):
      """Initialize sensor

      :param robot: Robot, to which this sensor belongs
      :param direction: direction, 2-tuple, of the sensor
      """
      self.robot = robot
      self.dir = direction

   def get_value_probabilities(self):
      """Return the probabilities of individual observations depending on robot position"""
      dist = self.robot.get_dist_to_wall(self.dir)
      p = {}
      p['n'] = max([1 - dist / self.DIST_OF_SURE_FAR, 0])
      p['f'] = 1 - p['n']
      return p

   def read(self):
      """Return a single sensor reading depending on robot position"""
      p = self.get_value_probabilities()
      return weighted_random_choice(p)


class Robot(HMM):
   """Robot in a maze as HMM"""

   def __init__(self, sensor_directions=None, move_probs=None):
      """Initialize robot with sensors and transition model

      :param sensor_directions: list of directions of individual sensors
      :param move_probs: distribution over move directions
      """
      self.maze = None
      self.position = None
      if not sensor_directions:
         sensor_directions = ALL_DIRS
      self.sensors = []
      for dir in sensor_directions:
         self.sensors.append(NearFarSensor(robot=self, direction=dir))
      self.active_sensors = []
      self.move_probs = move_probs if move_probs else FWD_MOVE_PROBS

   def observe(self, state=None):
      """Perform single observation of all sensors

      :param state: robot state (position) for which the observation
                    shall be made. If no state is given, the current
                    robot position is used.
      :return: tuple of individual sensor readings
      """
      if not state:
         # return tuple(s.read() for s in self.sensors)
         return tuple(self.sensors[i].read() for i in self.active_sensors)
      saved_pos = self.position
      self.position = state
      self.set_active_sensors()
      obs = self.observe()
      self.position = saved_pos
      self.set_active_sensors()
      return obs

   def next_move_dir(self):
      """Return the direction of next move"""
      return weighted_random_choice(self.move_probs)

   def get_dist_to_wall(self, dir, pos=None):
      """Return the distance to wall"""
      if not pos:
         pos = self.position
      pos = (pos[0], pos[1])
      return self.maze.get_dist_to_wall(pos, dir)

   def get_states(self):
      """Return the list of possible states"""
      states = []
      fps = self.maze.get_free_positions()
      for fp in fps:
         for ori in ALL_ORIS:
            states.append((fp[0], fp[1], ori))
      return states

   def get_targets(self, state):
      """Return the list of all states reachable in one step from the given state"""
      tgts = [state]
      forward_position = add((state[0], state[1]), ALL_DIRS[state[2]])
      if self.maze.is_free(forward_position):
         tgts.append((forward_position[0], forward_position[1], state[2]))
      tgts.append((state[0], state[1], (state[2] + 1) % 4))
      tgts.append((state[0], state[1], (state[2] - 1) % 4))
      return tgts

   def get_observations(self):
      """Return the list of all possible observations"""
      # sensor_domains = [s.VALUES for s in self.sensors]
      sensor_domains = [self.sensors[i].VALUES for i in self.active_sensors]
      return list(product(*sensor_domains))

   def get_next_state_distr(self, cur_state):
      """Return the distribution over possible next states

      Takes the walls around current state into account.
      """
      p = Counter()

      forward_position = add((cur_state[0], cur_state[1]), ALL_DIRS[cur_state[2]])
      if self.maze.is_free(forward_position):
         p[(forward_position[0], forward_position[1], cur_state[2])] = self.move_probs[FWD]

      p[(cur_state[0], cur_state[1], (cur_state[2] + 1) % 4)] = self.move_probs[TURN_RIGHT]
      p[(cur_state[0], cur_state[1], (cur_state[2] - 1) % 4)] = self.move_probs[TURN_LEFT]

      return normalized(p)

   def pt(self, cur_state, next_state):
      """Return a single transition probability"""
      p = self.get_next_state_distr(cur_state)
      return p[next_state]

   def pe(self, pos, obs):
      """Return the probability of observing obs in state pos"""
      # Store current robot position and set a new one
      stored_pos = self.position
      self.position = pos
      self.set_active_sensors()
      # Compute the probability of observation
      p = 1
      for sensor, value in zip([self.sensors[i] for i in self.active_sensors], obs):
         pd = sensor.get_value_probabilities()
         p *= pd[value]
      # Restore robot position
      self.position = stored_pos
      self.set_active_sensors()
      return p

   def assign_maze(self, maze):
      self.maze = maze

   def set_active_sensors(self):
      """Set a list of active sensors according to the orientation"""
      self.active_sensors = SENSORS_SWITCH[self.position[2]]

   def step(self, state):
      """Generate a next state for the current state"""
      next_pos = super().step(state)
      self.position = next_pos
      self.set_active_sensors()
      return next_pos

   def simulate(self, init_state=None, n_steps=5):
      """Perform several simulation steps starting from the given initial state

      :return: 2-tuple, sequence of states, and sequence of observations
      """
      if not init_state:
         init_state = self.position
      return super().simulate(init_state, n_steps)

   def init_models(self):
      """Initialize the trasition and emission models"""
      self.A = self._initialize_transition_model()
      self.B = self._initialize_emmision_model()


   def _initialize_transition_model(self):
      """Initialize transition model as a matrix of probability distribution"""
      model = ProbabilityMatrix(states=sorted(self.get_states()), obs=sorted(self.get_states()))
      for state in self.get_states():
         for successor, prob in self.get_next_state_distr(state).items():
            model[state, successor] = prob
      return model

   def _initialize_emmision_model(self):
      """Initialize emission model as a matrix of probability distribution"""
      model = ProbabilityMatrix(states=sorted(self.get_states()), obs=sorted(self.get_observations()))
      stored_state = self.position
      for state in self.get_states():
         self.position = state
         for obs in self.get_observations():
            prob = 1
            for sensor, value in zip(self.sensors, obs):
               p = sensor.get_value_probabilities()
               prob *= p[value]
            model[obs, state] = prob
      # Restore robot position
      self.position = stored_state
      return model
