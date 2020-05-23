"""
Implementation of Robot-in-a-maze domain as Hidden Markov Model.

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
Part of semestral project 2: Robot localization using HMM
"""

import random
import pandas as pd
import numpy as np
from collections import Counter
from itertools import product

from utils import weighted_random_choice, normalized
from hmm import HMM
from probability_vector import ProbabilityVector, ProbabilityMatrix


NORTH = (-1, 0)
EAST = (0, 1)
SOUTH = (1, 0)
WEST = (0, -1)
ALL_DIRS = (NORTH, EAST, SOUTH, WEST)

DEFAULT_MOVE_PROBS = {
    NORTH: 0.25,
    EAST: 0.25,
    SOUTH: 0.25,
    WEST: 0.25
}



def add(s, dir):
    """Add direction to a state"""
    return tuple(a+b for a, b in zip(s, dir))


def direction(s1, s2):
    """Return a direction vector from s1 to s2"""
    return tuple(b-a for a, b in zip(s1, s2))


def manhattan(s1, s2):
    return sum(abs(a) for a in direction(s1, s2))


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
        self.max_pos = (len(self.map)-2, len(self.map[0])-2)

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
        for r in range(1, self.height-1):
            for c in range(1, self.width-1):
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
        p['n'] = max([1 - dist/self.DIST_OF_SURE_FAR, 0])
        p['f'] = 1 - p['n']
        return p

    def read(self):
        """Return a single sensor reading depending on robot position"""
        p = self.get_value_probabilities()
        return weighted_random_choice(p)


class Robot(HMM):
    """Robot in a maze as HMM"""

    def __init__(self, sensor_directions=ALL_DIRS, move_probs=DEFAULT_MOVE_PROBS):
        """Initialize robot with sensors and transition model
        
        :param sensor_directions: list of directions of individual sensors
        :param move_probs: distribution over move directions 
        """
        self.maze = None
        self.position = None
        self.sensors = [NearFarSensor(robot=self, direction=dir) for dir in sensor_directions]
        self.move_probs = move_probs

    # @overrids
    def get_states(self):
        """Return the list of possible states"""
        return self.maze.get_free_positions()

    # @overrids
    def get_targets(self, state):
        """Return the list of all states reachable in one step from the given state"""
        tgts = [state]
        for dir in ALL_DIRS:
            next_state = add(state, dir)
            if not self.maze.is_free(next_state): continue
            tgts.append(next_state)
        return tgts

    # @overrids
    def get_observations(self):
        """Return the list of all possible observations"""
        sensor_domains = [s.VALUES for s in self.sensors]
        return list(product(*sensor_domains))

    # @overrids
    def step(self, state):
        """Generate a next state for the current state"""
        next_pos = super().step(state)
        # print(next_pos)
        self.position = next_pos
        return next_pos

    # @overrids
    def observe(self, state=None):
        """Perform single observation of all sensors
        
        :param state: robot state (position) for which the observation
                      shall be made. If no state is given, the current 
                      robot position is used.
        :return: tuple of individual sensor readings
        """
        if not state:
            return tuple(s.read() for s in self.sensors)
        saved_pos = self.position
        self.position = state
        obs = self.observe()
        self.position = saved_pos
        return obs

    # @overrids
    def simulate(self, init_state=None, n_steps=5):
        """Perform several simulation steps starting from the given initial state
        :return: 2-tuple, sequence of states, and sequence of observations
        """
        if not init_state:
            init_state = self.position
        return super().simulate(init_state, n_steps)

    # @overrids
    def pt(self, cur_state, next_state):
        """Return a single transition probability"""
        p = self._get_next_state_distr(cur_state)
        return p[next_state]

    # @overrids
    def pe(self, pos, obs):
        """Return the probability of observing obs in state pos"""
        # Store current robot position and set a new one
        stored_pos = self.position
        self.position = pos
        # Compute the probability of observation
        p = 1
        for sensor, value in zip(self.sensors, obs):
            pd = sensor.get_value_probabilities()
            p *= pd[value]
        # Restore robot position
        self.position = stored_pos
        return p

    def assign_maze(self, maze):
        self.maze = maze

    def init_models(self):
        self.A = self._initialize_transition_model()
        self.B = self._initialize_emmision_model()

    def set_random_position(self):
        """Set the robot to a random admissible state"""
        self.position = random.choice(self.maze.get_free_positions())

    def next_move_dir(self):
        """Return the direction of next move"""
        return weighted_random_choice(self.move_probs)

    def get_dist_to_wall(self, dir, pos=None):
        """Return the distance to wall"""
        if not pos:
            pos = self.position
        return self.maze.get_dist_to_wall(pos, dir)

    def _get_next_state_distr(self, cur_state):
        """Return the distribution over possible next states
        
        Takes the walls around current state into account.
        """
        p = Counter()
        for dir in ALL_DIRS:
            next_state = add(cur_state, dir)
            if not self.maze.is_free(next_state):
                pass
            else:
                p[next_state] = self.move_probs[dir]
        return normalized(p)

    def _initialize_transition_model(self):
        model = ProbabilityMatrix(states=sorted(self.get_states()), obs=sorted(self.get_states()))
        for state in self.get_states():
            for successor, prob in self._get_next_state_distr(state).items():
                model[state, successor] = prob
        return model

    def _initialize_emmision_model(self):
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


