"""
Hidden Markov model interface

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from utils import weighted_random_choice


class HMM:

    def get_states(self):
        """Return the list of hidden states for the domain"""
        raise NotImplementedError('get_states() must be overloaded in derived classes')

    def get_targets(self, state):
        """Return the list of states to which one can get from the current state

        The implementation depends on the transition probabilities.
        In the default implementation it returns the set of all the states.
        Can be overloaded to increase efficiency of some algorithms.
        """
        return self.get_states()

    def get_sources(self, state):
        """Return the list of states from which one can get to the given state

        The implementation depends on the transition probabilities.
        In the default implementation it returns the set of all the states.
        Can be overloaded to increase efficiency of some algorithms.
        """
        return self.get_states()

    def get_observations(self):
        """Return the list of observations for the domain"""
        raise NotImplementedError('get_observations() must be overloaded in derived classes')

    def pt(self, cur_state, next_state):
        """Return the probability of transition from current to next state"""
        raise NotImplementedError('pt() must be overloaded in derived classes')

    def pe(self, state, obs):
        """Return the probability of the given observation when in current state"""
        raise NotImplementedError('pe() must be overloaded in derived classes')

    def step(self, state):
        """Make a time step: generate a particular next state for the current state"""
        # Compile a distribution over next states
        next_dist = {next_state: self.pt(state, next_state)
                     for next_state in self.get_targets(state)}
        # Sample from the distribution
        return weighted_random_choice(next_dist)

    def observe(self, state):
        """Generate a particulat observation for the current state"""
        # Compile a distribution over observations in current state
        obs_dist = {obs: self.pe(state, obs)
                    for obs in self.get_observations()}
        # Sample from the distribution
        return weighted_random_choice(obs_dist)

    def simulate(self, init_state, n_steps):
        """Perform several simulation steps starting from the given initial state

        :return: 2-tuple, sequence of states, and sequence of observations
        """
        last_state = init_state
        states, observations = [], []
        for i in range(n_steps):
            state = self.step(last_state)
            observation = self.observe(state)
            states.append(state)
            observations.append(observation)
            last_state = state
        return states, observations