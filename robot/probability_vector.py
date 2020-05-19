from collections import Counter

import numpy as np
import pandas as pd


class ProbabilityVector:
    def __init__(self, data):
        self.states = sorted(data.keys())
        self.values = np.array(list(map(lambda x: data[x], self.states))).reshape(-1, 1)
        # if self.values.shape and self.values.shape[0] != len(self.states):
        #     self.values.reshape(-1, 1)

    @classmethod
    def rand_initialize(cls, states):
        rand = np.random.rand(len(states))
        rand = [k * 1/sum(rand) for k in rand]
        prob_vector = cls(dict(zip(states, rand)))
        return prob_vector

    def normalize(self):
        s = sum(self.values)
        factor = 1 / s
        self.values = [val * factor for val in self.values]

    def to_data_frame(self):
        return pd.Series(np.squeeze(self.values), index=self.states)

    def __repr__(self):
        return "p({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if isinstance(other, ProbabilityVector):
            if (self.states == other.states) and (self.values == other.values).all():
                return True
        elif isinstance(other, Counter):
            if (self.states == sorted(other.keys())) \
                    and all([abs(self.values[i] - other[state]) < 1e-12 for i, state in enumerate(self.states)]):
                return True
        else:
            raise NotImplementedError
        return False

    def __getitem__(self, state):
        if state not in self.states:
            raise ValueError("Unknown state")
        index = self.states.index(state)
        return float(self.values[0, index])


    def __mul__(self, other):
        # multiplication with ProbabilityVector column
        if isinstance(other, ProbabilityVector):
            vals = self.values * other.values
        # multiplication with Dataframe column
        elif isinstance(other, pd.Series):
            vals = self.values * other.values.reshape(-1, 1)
        # multiplication with numpy ndarray column
        elif isinstance(other, np.ndarray):
            vals = self.values * other.reshape(-1, 1)
        # element-wise matrix multiplication with Pandas matrix
        elif isinstance(other, pd.DataFrame):
            return self.to_data_frame() * other
        else:
            raise NotImplementedError

        return ProbabilityVector(dict(zip(self.states, vals)))

    def __rmatmul__(self, other):
        # matrix multiplication with numpy ndarray matrix
        if isinstance(other, np.ndarray):
            return ProbabilityVector(dict(zip(self.states, other @ self.values)))
        # matrix multiplication with DataFrame matrix
        elif isinstance(other, pd.DataFrame):
            return ProbabilityVector(dict(zip(self.states, other.values @ self.values)))

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]