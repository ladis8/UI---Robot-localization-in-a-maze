"""
Functions for inference in HMMs

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from collections import Counter
from utils import normalized
import numpy as np

def update_belief_by_time_step(prev_B, hmm):
   """Update the distribution over states by 1 time step.

   :param prev_B: Counter, previous belief distribution over states
   :param hmm: contains the transition model hmm.pt(from,to)
   :return: Counter, current (updated) belief distribution over states
   """
   cur_B = Counter()
   # Your code here
   # raise NotImplementedError('You must implement update_belief_by_time_step()')
   for Xt1 in hmm.get_states():
      cur_B[Xt1] = 0
      for Xt0 in hmm.get_states():
         cur_B[Xt1] += hmm.pt(Xt0, Xt1) * prev_B[Xt0]
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
   B = prior  # This shall be iteratively updated
   Bs = []  # This shall be a collection of Bs over time steps
   # Your code here
   # raise NotImplementedError('You must implement predict()')
   for i in range(n_steps):
      B = update_belief_by_time_step(B, hmm)
      Bs.append(B)
   return Bs


def update_belief_by_evidence(prev_B, e, hmm, normalize=False):
   """Update the belief distribution over states by observation

   :param prev_B: Counter, previous belief distribution over states
   :param e: a single evidence/observation used for update
   :param hmm: HMM for which we compute the update
   :param normalize: bool, whether the result shall be normalized
   :return: Counter, current (updated) belief distribution over states
   """
   # Create a new copy of the current belief state
   cur_B = Counter(prev_B)
   # Your code here
   # raise NotImplementedError('You must implement update_belief_by_evidence()')
   for Xt in hmm.get_states():
      cur_B[Xt] = hmm.pe(Xt, e) * prev_B[Xt]
   return cur_B


def forward1(prev_f, cur_e, hmm, normalize=False):
   """Perform a single update of the forward message

   :param prev_f: Counter, previous belief distribution over states
   :param cur_e: a single current observation
   :param hmm: HMM, contains the transition and emission models
   :param normalize: bool, should we normalize on the fly?
   :return: Counter, current belief distribution over states
   """
   # Your code here
   # raise NotImplementedError('You must implement forward1()')
   cur_f = update_belief_by_time_step(prev_f, hmm)
   cur_f = update_belief_by_evidence(cur_f, cur_e, hmm)
   if normalize:
      cur_f = normalized(cur_f)
   return cur_f


def forward(init_f, e_seq, hmm):
   """Compute the filtered belief states given the observation sequence

   :param init_f: Counter, initial belief distribution over the states
   :param e_seq: sequence of observations
   :param hmm: contains the transition and emission models
   :return: sequence of Counters, estimates of belief states for all time slices
   """
   f = init_f  # Forward message, updated each iteration
   fs = []  # Sequence of forward messages, one for each time slice
   # Your code here
   # raise NotImplementedError('You must implement forward()')
   for e in e_seq:
      f = forward1(f, e, hmm, True)
      fs.append(f)
   return fs


def likelihood(prior, e_seq, hmm):
   """Compute the likelihood of the model wrt the evidence sequence

   In other words, compute the marginal probability of the evidence sequence.
   :param prior: Counter, initial belief distribution over states
   :param e_seq: sequence of observations
   :param hmm: contains the transition and emission models
   :return: number, likelihood
   """
   # Your code here
   # raise NotImplementedError('You must implement likelihood()')
   f = prior
   for e in e_seq:
      f = forward1(f, e, hmm)
   lhood = sum(f.values())
   return lhood


def backward1(next_b, next_e, hmm):
   """Propagate the backward message

   :param next_b: Counter, the backward message from the next time slice
   :param next_e: a single evidence for the next time slice
   :param hmm: HMM, contains the transition and emission models
   :return: Counter, current backward message
   """
   cur_b = Counter()
   # Your coude here
   # raise NotImplementedError('You must implement backward1()')
   for Xt0 in hmm.get_states():
      cur_b[Xt0] = 0
      for Xt1 in hmm.get_states():
         cur_b[Xt0] += hmm.pe(Xt1, next_e) * hmm.pt(Xt0, Xt1) * next_b[Xt1]
   return cur_b


def forwardbackward(priors, e_seq, hmm):
   """Compute the smoothed belief states given the observation sequence

   :param priors: Counter, initial belief distribution over rge states
   :param e_seq: sequence of observations
   :param hmm: HMM, contians the transition and emission models
   :return: sequence of Counters, estimates of belief states for all time slices
   """
   se = []  # Smoothed belief distributions
   # Your code here
   # raise NotImplementedError('You must implement forwardbackward()')
   fs = forward(priors, e_seq, hmm)
   b = Counter()
   for Xt in hmm.get_states():
      b[Xt] = 1
   for e, f in zip(reversed(e_seq), reversed(fs)):
      s = Counter()
      for Xt in hmm.get_states():
         s[Xt] = f[Xt] * b[Xt]
      se.insert(0, normalized(s))
      b = backward1(b, e, hmm)
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
   cur_m = Counter()  # Current (updated) max message
   predecessors = {}  # The best of previous states for each current state
   # Your code here
   # raise NotImplementedError('You must implement viterbi1()')
   for cur_X in hmm.get_states():
      ptm_max = 0
      for X in hmm.get_states():
         ptm = hmm.pt(X, cur_X) * prev_m[X]
         if ptm > ptm_max:
            ptm_max = ptm
            predecessors[cur_X] = X
      cur_m[cur_X] = hmm.pe(cur_X, cur_e) * ptm_max
   return cur_m, predecessors


def viterbi(priors, e_seq, hmm):
   """Find the most likely sequence of states using Viterbi algorithm

   :param priors: Counter, prior belief distribution
   :param e_seq: sequence of observations
   :param hmm: HMM, contains the transition and emission models
   :return: (sequence of states, sequence of max messages)
   """
   ml_seq = []  # Most likely sequence of states
   ms = []  # Sequence of max messages
   # Your code here
   # raise NotImplementedError('You must implement viterbi()')
   prev_m = forward1(priors, e_seq[0], hmm)
   ms.append(prev_m)
   predecessors_seq = []
   for e in e_seq[1:]:
      cur_m, predecessors = viterbi1(prev_m, e, hmm)
      ms.append(cur_m)
      prev_m = cur_m
      predecessors_seq.append(predecessors)
   cur_p = ms[-1].most_common(1)[0][0]
   ml_seq.append(cur_p)
   for p in reversed(predecessors_seq):
      cur_p = p[cur_p]
      ml_seq.insert(0, cur_p)
   return ml_seq, ms


def viterbi1_log(prev_log_m, cur_e, hmm):
   """Perform a single update of the max message for Viterbi algorithm
      using log probabilities

   :param prev_m: Counter, max message from the previous time slice
   :param cur_e: current observation used for update
   :param hmm: HMM, contains transition and emission models
   :return: (cur_m, predecessors), i.e.
            Counter, an updated max message, and
            dict with the best predecessor of each state
   """
   cur_log_m = Counter()  # Current (updated) max message
   predecessors = {}  # The best of previous states for each current state
   # Your code here
   # raise NotImplementedError('You must implement viterbi1()')
   for cur_X in hmm.get_states():
      ptm_max = -np.inf
      for X in hmm.get_states():
         ptm = np.log(hmm.pt(X, cur_X)) + prev_log_m[X]
         if ptm > ptm_max:
            ptm_max = ptm
            predecessors[cur_X] = X
      cur_log_m[cur_X] = np.log(hmm.pe(cur_X, cur_e)) + ptm_max # (105b) recursion step
   return cur_log_m, predecessors


def viterbi_log(priors, e_seq, hmm):
   """Find the most likely sequence of states using Viterbi algorithm
      using log probabilities

      :param priors: Counter, prior belief distribution
      :param e_seq: sequence of observations
      :param hmm: HMM, contains the transition and emission models
      :return: (sequence of states, sequence of max messages)
      """
   ml_seq = []  # Most likely sequence of states
   log_ms = []  # Sequence of max messages
   # Your code here
   # raise NotImplementedError('You must implement viterbi()')
   np.seterr(divide='ignore')
   # forward1 begin
   pi_log = Counter()
   for Xt1 in hmm.get_states():
      pi_log[Xt1] = 0
      for Xt0 in hmm.get_states():
         if hmm.pt(Xt0, Xt1) * priors[Xt0] > 0:
            pi_log[Xt1] += np.log(hmm.pt(Xt0, Xt1) * priors[Xt0])
   b_log = Counter()
   for Xt in hmm.get_states():
      if hmm.pe(Xt, e_seq[0]) > 0 or 1:
         b_log[Xt] = np.log(hmm.pe(Xt, e_seq[0]))
   prev_log_m = Counter()
   for Xt in hmm.get_states():
      prev_log_m[Xt] = pi_log[Xt] + b_log[Xt] # (105a) initial set
   # forward1 end
   log_ms.append(prev_log_m)
   predecessors_seq = []
   for e in e_seq[1:]:
      cur_log_m, predecessors = viterbi1_log(prev_log_m, e, hmm)
      log_ms.append(cur_log_m)
      prev_log_m = cur_log_m
      predecessors_seq.append(predecessors)
   cur_p = log_ms[-1].most_common(1)[0][0] # (105c) termination step
   ml_seq.append(cur_p)
   for p in reversed(predecessors_seq):
      cur_p = p[cur_p]
      ml_seq.insert(0, cur_p)
   return ml_seq, log_ms