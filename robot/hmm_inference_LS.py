"""
Functions for inference in HMMs

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from collections import Counter
from utils import normalized


def update_belief_by_time_step(prev_B, hmm):
    """Update the distribution over states by 1 time step.

    :param prev_B: Counter, previous belief distribution over states
    :param hmm: contains the transition model hmm.pt(from,to)
    :return: Counter, current (updated) belief distribution over states
    """
    cur_B = Counter()
    for cur_state in hmm.get_states():
        for prev_state in hmm.get_states():
            cur_B[cur_state] += hmm.T[prev_state][cur_state] * prev_B[prev_state]
    return cur_B


def predict(n_steps, prior, hmm):
    """Predict belief state n_steps to the future

    :param n_steps: number of time-step updates we shall execute
    :param prior: Counter, initial distribution over the states
    :param hmm: contains the transition model hmm.pt(from, to)
    :return: sequence of belief distributions (list of Counters),
             for each time slice one belief distribution;
             prior distribution shall not be included
    """
    B_t = prior  # This shall be iteratively updated
    B_all = []    # This shall be a collection of Bs over time steps
    for t in range(n_steps):
        B_t = update_belief_by_time_step(B_t, hmm)
        B_all.append(B_t)
    return B_all


def update_belief_by_evidence(prev_B, e, hmm):
    """Update the belief distribution over states by observation

    :param prev_B: Counter, previous belief distribution over states
    :param e: a single evidence/observation used for update
    :param hmm: HMM for which we compute the update
    :return: Counter, current (updated) belief distribution over states
    """
    cur_B = Counter()
    for cur_state in hmm.get_states():
        cur_B[cur_state] = hmm.E[cur_state][e] * prev_B[cur_state]
    return cur_B


def forward1(prev_f, cur_e, hmm):
    """Perform a single update of the forward message

    :param prev_f: Counter, previous belief distribution over states
    :param cur_e: a single current observation
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current belief distribution over states
    """
    # update by time
    cur_f = update_belief_by_time_step(prev_f, hmm)
    # update by evidence
    cur_f = update_belief_by_evidence(cur_f, cur_e, hmm)
    return cur_f


def forward(init_f, e_seq, hmm):
    """Compute the filtered belief states given the observation sequence

    :param init_f: Counter, initial belief distribution over the states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: sequence of Counters, i.e., estimates of belief states for all time slices
    """
    cur_f = init_f    # Forward message, updated each iteration
    fs = []       # Sequence of forward messages, one for each time slice
    for cur_e in e_seq:
        cur_f = normalized(forward1(cur_f, cur_e, hmm))
        fs.append(cur_f)
    return fs


def likelihood(prior, e_seq, hmm):
    """Compute the likelihood of the model wrt the evidence sequence

    In other words, compute the marginal probability of the evidence sequence.
    :param prior: Counter, initial belief distribution over states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: number, likelihood
    """
    cur_l = prior
    for cur_e in e_seq:
        cur_l = forward1(cur_l, cur_e, hmm)

    lhood = sum([cur_l[state] for state in hmm.get_states()])
    return lhood



def backward1(next_b, next_e, hmm):
    """Propagate the backward message

    :param next_b: Counter, the backward message from the next time slice
    :param next_e: a single evidence for the next time slice
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current backward message
    """
    cur_b = Counter()
    for next_state in hmm.get_states():
        for cur_state in hmm.get_states():
            cur_b[cur_state] += hmm.E[next_state][next_e] * next_b[next_state] * hmm.T[cur_state][next_state]
    return cur_b


def forwardbackward(prior, e_seq, hmm):
    """Compute the smoothed belief states given the observation sequence

    :param prior: Counter, initial belief distribution over rge states
    :param e_seq: sequence of observations
    :param hmm: HMM, contians the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    se = forward(prior, e_seq, hmm)
    bt = Counter({state: 1.0 for state in hmm.get_states()})
    for t in range(len(se) - 1, -1, -1):
        ft = se[t]
        se[t] = normalized(Counter({state: ft[state] * bt[state] for state in hmm.get_states()}))
        bt = backward1(bt, e_seq[t], hmm)
    return se


def viterbi1(prev_m, cur_e, hmm):
    """Perform a single update of the max message for Viterbi algorithm

    :param prev_m: Counter, max message from the previous time slice
    :param cur_e: current observation used for update
    :param hmm: HMM, contains transition and emission models
    :return: (cur_m, predecessors), i.e.
             Counter, an updated max message, and
             dict with the best predecessor of each state
    """
    cur_m = Counter()   # Current (updated) max message
    predecessors = {}   # The best of previous states for each current state
    for cur_state in hmm.get_states():
        prev_states = [(prev_state, hmm.T[prev_state][cur_state] * prev_m[prev_state])
                       for prev_state in hmm.get_states()]
        best_prev_state, p_max = max(prev_states, key=lambda x: x[1])
        cur_m[cur_state] = hmm.E[cur_state][cur_e] * p_max
        predecessors[cur_state] = best_prev_state
    return cur_m, predecessors


def viterbi(prior, e_seq, hmm):
    """Find the most likely sequence of states using Viterbi algorithm

    :param prior: Counter, prior belief distribution
    :param e_seq: sequence of observations
    :param hmm: HMM, contains the transition and emission models
    :return: (sequence of states, sequence of max messages)
    """
    ms_best_states = []  # Most likely sequence of states
    ms, ms_pred = [], []      # Sequence of max messages
    ms.append(forward1(prior, e_seq[0], hmm))
    for cur_e in e_seq[1:]:
        ms_t, ms_pred_t = viterbi1(ms[-1], cur_e, hmm)
        ms.append(ms_t)
        ms_pred.append(ms_pred_t)


    # get best state for last element in ms
    ms_last = ms[-1]
    ms_best_state_last = max(ms_last, key=ms_last.get)
    ms_best_states.append(ms_best_state_last)
    # iterate over best predessessors until last element
    for ms_pred_t in ms_pred[::-1]:
        ms_best_state_last = ms_pred_t[ms_best_state_last]
        ms_best_states.append(ms_best_state_last)

    return ms_best_states.reverse(), ms

