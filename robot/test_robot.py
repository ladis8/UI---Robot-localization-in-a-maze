"""
Example scripts for Robot in a maze HMM

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

# from hmm_inference import *
import os
import numpy as np
import random as rnd

USE_MATRIX = False
# ROBOT_MODEL = 'NESW'

if USE_MATRIX:
    from hmm_inference_matrix import *
else:
    from hmm_inference import *

from robot import *
from utils import normalized, init_belief, get_key_value_tuples
import probability_vector as pv
import copy
import numpy as np


direction_probabilities = {
    NORTH: 0.25,
    EAST: 0.25,
    SOUTH: 0.25,
    WEST: 0.25
}

#TODO: move more functions to tools

def init_maze(maze_name='mazes/rect_6x10_sparse.map'):
    """Create and initialize robot instance for subsequent test"""
    m = Maze(maze_name)
    #    robot = Robot(ALL_DIRS, direction_probabilities)
    robot = Robot()
    robot.assign_maze(m)
    robot.position = (1, 1)
    # print_robot(robot)
    # print('Robot at ', robot.position)
    return robot

def print_robot(robot):
    pos = robot.position
    m = robot.maze.map
    for i in range(len(m)):
        for ii in range(len(m[0])):
            if pos == (i, ii):
                print('R', end='')
                continue
            print(m[i][ii], end='')
            print('\n', end='')


def test_pt():
    """Try to compute transition probabilities for certain position"""
    robot = init_maze()
    robot.position = (2, 10)
    print('Robot at', robot.position)
    for pos in robot.maze.get_free_positions():
        p = robot.pt(robot.position, pos)
        if p > 0:
            print('Prob of transition to', pos, 'is', p)


def test_pe():
    """Try to compute the observation probabilities for certain position"""
    robot = init_maze()
    robot.position = (1, 5)
    print('Robot at', robot.position)
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


def test_filtering():
    """Try to run filtering for robot domain"""

    robot = init_maze()
    if USE_MATRIX: robot.init_models()
    states, obs = robot.simulate(n_steps=10)
    print('Running filtering...')

    initial_belief = init_belief(robot.get_states(), USE_MATRIX)
    beliefs = forward(initial_belief, obs, robot)

    for state, belief in zip(states, beliefs):
        print('Real state:', state)
        print('Sorted beliefs:')
        key_value_tuples = get_key_value_tuples(belief, USE_MATRIX)

        for k, v in sorted(sorted(key_value_tuples), key=lambda x: x[1], reverse=True):
            if v > 0:
                print(k, ':', v)


def test_smoothing():
    """Try to run smoothing for robot domain"""
    robot = init_maze()
    if USE_MATRIX: robot.init_models()
    states, obs = robot.simulate(init_state=(1, 10), n_steps=10)
    print('Running smoothing...')

    initial_belief = init_belief(robot.get_states(), USE_MATRIX)
    beliefs = forwardbackward(initial_belief, obs, robot)

    for state, belief in zip(states, beliefs):
        print('Real state:', state)
        print('Sorted beliefs:')
        key_value_tuples = get_key_value_tuples(belief, USE_MATRIX)
        for k, v in sorted(key_value_tuples, key=lambda x: x[1], reverse=True):
            if v > 0:
                print(k, ':', v)


def test_viterbi():
    """Try to run Viterbi alg. for robot domain"""
    robot = init_maze()
    if USE_MATRIX: robot.init_models()
    states, obs = robot.simulate(init_state=(3, 3), n_steps=10)

    print('Running Viterbi...')
    initial_belief = init_belief(robot.get_states(), USE_MATRIX)
    ml_states, max_msgs = viterbi(initial_belief, obs, robot)
    for real, est in zip(states, ml_states):
        print('Real pos:', real, '| ML Estimate:', est)


def test_viterbi_log():
    """Try to run log-based Viterbi alg. for robot domain"""
    robot = init_maze()
    if USE_MATRIX: robot.init_models()
    states, obs = robot.simulate(init_state=(1, 1), n_steps=10)

    print('Running log-based Viterbi...')
    initial_belief = init_belief(robot.get_states(), USE_MATRIX)

    ml_states, max_msgs = viterbi(initial_belief, obs, robot, underflow_prevention=True)
    iter = 0
    for real, est in zip(states, ml_states):
        iter += 1
        print('Iter:', iter, '| Real pos:', real, '| ML Estimate:', est)


# TODO: test matrixes by simulating
def test_matrix_alg_steps():
    def test_steps_normal(robot, obs, obs2, initial_belief):
        from hmm_inference import forward1, backward1, viterbi1, viterbi1_log

        v1 = forward1(initial_belief, obs, robot)
        v2 = forward1(v1, obs2, robot)
        back_v = backward1(v2, obs2, robot)
        viterbi_v_log = viterbi1_log(initial_belief, obs, robot)[0]
        viterbi_v = viterbi1(initial_belief, obs, robot)[0]
        return [v1, v2, back_v, viterbi_v, viterbi_v_log]

    def test_steps_matrix(robot, obs, obs2, initial_belief):
        from hmm_inference_matrix import forward1, backward1, viterbi1, viterbi1_log
        initial_belief = pv.ProbabilityVector.initialize_from_dict(initial_belief)

        v1_m = forward1(initial_belief, obs, robot)
        v2_m = forward1(v1_m, obs2, robot)
        back_v_m = backward1(v2_m, obs2, robot)
        viterbi_v_m_log = viterbi1_log(initial_belief, obs, robot)[0]
        viterbi_v_m = viterbi1(initial_belief, obs, robot)[0]
        return [v1_m, v2_m, back_v_m, viterbi_v_m, viterbi_v_m_log]

    robot = init_maze()
    robot.position = (1, 5)
    robot.init_models()
    print(robot.A)

    states, obs = robot.simulate(init_state=(3, 3), n_steps=20)

    obs = ('f', 'n', 'f', 'n')
    obs2 = ('n', 'n', 'f', 'f')
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})

    for v_normal, v_matrix in zip(
            test_steps_normal(robot, obs, obs2, initial_belief),
            test_steps_matrix(robot, obs, obs2, initial_belief)
            ):
        print(v_normal == v_matrix)
        # for state in v_matrix.states:
        #     print(state, ": "," ", v_matrix[state], "/", v_normal[state])
        print()


def get_hit_rate(states, beliefs):
    hits = 0
    for s, b in zip(states, beliefs):
        if s == b:
            hits += 1
    return hits / len(states)


def get_manhattan_dist(states, beliefs):
    sum = 0
    for s, b in zip(states, beliefs):
        sum += abs((s[0] - b[0]) + (s[1] - b[1]))
    return sum / len(states)


def get_euclidean_dist(states, beliefs):
    sum = 0
    for s, b in zip(states, beliefs):
        sum += np.sqrt((s[0] - b[0]) ** 2 + (s[1] - b[1]) ** 2)
    return sum / len(states)


def evaluate1(steps=10, maze_name='mazes/rect_6x10_obstacles.map', file_obj=None, VERBOSE=2):

    hit_rate_filter, manhattan_dist_filter, euclidean_dist_filter = 0,0,0
    hit_rate_smooth, manhattan_dist_smooth, euclidean_dist_smooth = 0,0,0
    hit_rate_viterbi, manhattan_dist_viterbi, euclidean_dist_viterbi = 0,0,0

    robot = init_maze(maze_name)

    fp = robot.maze.get_free_positions()
    for ipos in fp:
        if VERBOSE >= 1: print('------------------------------------')
        if VERBOSE >= 1: print(np.round(100*(fp.index(ipos)+1)/len(fp), 0), '%\tof inital positions')
        robot.position = ipos

        states, obs = robot.simulate(n_steps=steps)
        initial_belief = normalized({pos: 1 for pos in robot.get_states()})

        # FILTERING
        if VERBOSE >= 2: print('Running filtering...')
        beliefs = forward(initial_belief, obs, copy.deepcopy(robot))
        most_beliefs = []
        for state, belief in zip(states, beliefs):
            most_belief_state = sorted(belief.items(), key=lambda x: x[1], reverse=True)[0][0]
            most_beliefs.append(most_belief_state)
            if VERBOSE >= 4: print('Real state:', state, '| Best belief:', most_belief_state)
        hit_rate_filter += get_hit_rate(states, most_beliefs)
        manhattan_dist_filter += get_manhattan_dist(states, most_beliefs)
        euclidean_dist_filter += get_euclidean_dist(states, most_beliefs)
        if VERBOSE >= 3: print('hit rate =', hit_rate_filter, '\nmanhattan dist =', manhattan_dist_filter, '\neuclidean dist =', euclidean_dist_filter)

        # SMOOTHING
        if VERBOSE >= 2: print('Running smoothing...')
        beliefs = forwardbackward(initial_belief, obs, copy.deepcopy(robot))
        most_beliefs = []
        for state, belief in zip(states, beliefs):
            most_belief_state = sorted(belief.items(), key=lambda x: x[1], reverse=True)[0][0]
            most_beliefs.append(most_belief_state)
            if VERBOSE >= 4: print('Real state:', state, '| Best belief:', most_belief_state)
        hit_rate_smooth += get_hit_rate(states, most_beliefs)
        manhattan_dist_smooth += get_manhattan_dist(states, most_beliefs)
        euclidean_dist_smooth += get_euclidean_dist(states, most_beliefs)
        if VERBOSE >= 3: print('hit rate =', hit_rate_smooth, '\nmanhattan dist =', manhattan_dist_smooth, '\neuclidean dist =', euclidean_dist_smooth)

        # VITERBI
        if VERBOSE >= 2: print('Running Viterbi...')
        ml_states, max_msgs = viterbi(initial_belief, obs, copy.deepcopy(robot))
        if VERBOSE >= 4:
            for real, est in zip(states, ml_states):
                print('Real pos:', real, '| ML Estimate:', est)
        hit_rate_viterbi += get_hit_rate(states, ml_states)
        manhattan_dist_viterbi += get_manhattan_dist(states, ml_states)
        euclidean_dist_viterbi += get_euclidean_dist(states, ml_states)
        if VERBOSE >= 3: print('hit rate =', hit_rate_viterbi, '\nmanhattan dist =', manhattan_dist_viterbi, '\neuclidean dist =', euclidean_dist_viterbi)

    if VERBOSE >= 2: print('------------------------------------')
    if VERBOSE >= 2: print('------------------------------------')

    hit_rate_filter /= len(fp)
    manhattan_dist_filter /= len(fp)
    euclidean_dist_filter /= len(fp)

    hit_rate_smooth /= len(fp)
    manhattan_dist_smooth /= len(fp)
    euclidean_dist_smooth /= len(fp)

    hit_rate_viterbi /= len(fp)
    manhattan_dist_viterbi /= len(fp)
    euclidean_dist_viterbi /= len(fp)

    if VERBOSE >= 2:
        print('hit rate =', hit_rate_filter, '\nmanhattan dist =', manhattan_dist_filter, '\neuclidean dist =', euclidean_dist_filter)
        print('hit rate =', hit_rate_smooth, '\nmanhattan dist =', manhattan_dist_smooth, '\neuclidean dist =', euclidean_dist_smooth)
        print('hit rate =', hit_rate_viterbi, '\nmanhattan dist =', manhattan_dist_viterbi, '\neuclidean dist =', euclidean_dist_viterbi)

    if file_obj:
        file_obj.write(str(hit_rate_filter) + '\t' + str(manhattan_dist_filter) + '\t' + str(euclidean_dist_filter) + '\n')
        file_obj.write(str(hit_rate_smooth) + '\t' + str(manhattan_dist_smooth) + '\t' + str(euclidean_dist_smooth) + '\n')
        file_obj.write(str(hit_rate_viterbi) + '\t' + str(manhattan_dist_viterbi) + '\t' + str(euclidean_dist_viterbi) + '\n')

def evaluate(steps=10):
    path = 'mazes/'
    mazes = os.listdir(path)
    # file = open("outfile.txt", "a+")

    # file.write('\n\n------------------------' + str(steps) + '\n')

    for maze in mazes[4:]:
        print('---', maze, '---')
        # file.write(maze + '\n')

        # (hit_rate_filter, manhattan_dist_filter, euclidean_dist_filter,
        #  hit_rate_smooth, manhattan_dist_smooth, euclidean_dist_smooth,
        #  hit_rate_viterbi, manhattan_dist_viterbi, euclidean_dist_viterbi) = evaluate1(steps=steps, maze_name=path+maze, file_object=file)

        # evaluate1(steps=steps, maze_name=path+maze, file_obj=file, VERBOSE=2)
        evaluate1(steps=steps, maze_name=path+maze)

        break

        # file.write(str(hit_rate_filter) + '\t' + str(manhattan_dist_filter) + '\t' + str(euclidean_dist_filter) + '\n')
        # file.write(str(hit_rate_smooth) + '\t' + str(manhattan_dist_smooth) + '\t' + str(euclidean_dist_smooth) + '\n')
        # file.write(str(hit_rate_viterbi) + '\t' + str(manhattan_dist_viterbi) + '\t' + str(euclidean_dist_viterbi) + '\n')

    # file.close()


if __name__ == '__main__':
    rnd.seed(314)
    print('Uncomment some of the tests in the main section')

    steps_list = [10, 20, 50, 100, 300]
    for st in steps_list:
        print('======', st, 'STEPS ======')
        evaluate(steps=st)
        print('')
        break


    #test_matrix_alg_steps()
    #test_pt()
    #test_pe()
    #test_simulate()
    # test_filtering()
    # test_smoothing()
    # test_viterbi()
    # test_viterbi_log()
