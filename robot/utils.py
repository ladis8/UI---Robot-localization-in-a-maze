"""
Helper functions for HMM project

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""


import itertools
import bisect
import random
from collections import Counter

import probability_vector as pv


def weighted_random_choice(pdist):
    """Choose an element from distribution given as dict of values and their probs."""
    weighted_choices = [(k,v) for k,v in pdist.items()]
    choices, weights = zip(*weighted_choices)
    cumdist = list(itertools.accumulate(weights))
    x = random.random() * cumdist[-1]
    return choices[bisect.bisect(cumdist, x)]


def test_weighted_random_choices():
    from collections import Counter
    pdist = {'a': 0.5, 'b': 0.3, 'c': 0.2}
    c = Counter()
    for i in range(100000):
        char = weighted_random_choice(pdist)
        c[char] += 1
    print(c)


def normalized(P, factor=None, return_normalization_factor=False):
    """Return a normalized copy of the distribution given as a Counter"""
    if not factor:
        s = sum(P.values())
        factor = 1 / s
    norm = Counter({k: factor*v for k, v in P.items()})
    if return_normalization_factor:
        return norm, factor
    else:
        return norm

def init_belief(states, use_probability_vector=False):
    """Initialize the uniform distribution either as Counter or Probability Vector"""

    if use_probability_vector:
        initial_belief = pv.ProbabilityVector.uniform_initialize(states)
    else:
        initial_belief = normalized({pos: 1 for pos in states})

    return initial_belief


def get_key_value_tuples(data, use_probability_vector=False):
    """Iterates over pairs (state, probability)"""
    if use_probability_vector:
        key_value_tuples = zip(data.states, data.values)
    else:
        key_value_tuples = data.items()
    return key_value_tuples


if __name__ == "__main__":
    test_weighted_random_choices()


