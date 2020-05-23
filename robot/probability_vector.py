from collections import Counter

import numpy as np
import pandas as pd


class ProbabilityVector:
    def __init__(self, states, values, sort=False):
        assert (len(states) == len(values))
        if sort:
            sorted_list = zip(*sorted(zip(states, values)))
            state, values = [list(x) for x in sorted_list]
        self.states = states
        self.values = values.reshape(-1, 1)

    @classmethod
    def initialize_from_dict(cls, vector_dict):
        states = sorted(vector_dict.keys())
        values = np.array(list(map(lambda state: vector_dict[state], vector_dict.keys())))
        return cls(states, values, sort=False)

    @classmethod
    def rand_initialize(cls, states):
        rand = np.random.rand(len(states))
        factor = 1/sum(rand)
        rand *= factor
        return cls(sorted(states), rand, sort=False)

    @classmethod
    def uniform_initialize(cls, states):
        values = np.ones(len(states)) * 1/len(states)
        return cls(sorted(states), values, sort=False)

    def normalize(self):
        factor = 1 / sum(self.values)
        self.values *= factor
        return self

    @property
    def log(self):
        #values = np.ma.log(self.values).filled(0)
        values = np.log(self.values)
        return ProbabilityVector(self.states, values)

    @property
    def df(self):
        return pd.Series(np.squeeze(self.values), index=self.states)

    @property
    def dict(self):
        return {state: val for state, val in zip(self.states, self.values[:, 0])}

    def __repr__(self):
        return "Probability Vector states: {} -> values {}.".format(self.states, self.values)

    def __eq__(self, other):
        if isinstance(other, ProbabilityVector):
            if (self.states == other.states) and (self.values == other.values).all():
                return True
        elif isinstance(other, Counter):
            # print([(np.isinf(self.values[i, 0]) and np.isinf(other[state])) or abs(self.values[i, 0] - other[state]) < 1e-12 for i, state in enumerate(self.states)])
            if (self.states == sorted(other.keys()) and \
                    all([(np.isinf(self.values[i, 0]) and np.isinf(other[state]))
                         or abs(self.values[i, 0] - other[state]) < 1e-12 for i, state in enumerate(self.states)])):
                return True
        else:
            raise NotImplementedError
        return False

    def __getitem__(self, state):
        if state not in self.states:
            raise ValueError("Unknown state")
        index = self.states.index(state)
        return self.values[index,0]

    def __add__(self, other):
        if isinstance(other, ProbabilityVector):
            values = self.values + other.values
            return ProbabilityVector(self.states, values)
        else:
            raise NotImplementedError

    # assume that states are same
    def __mul__(self, other):
        # multiplication with ProbabilityVector column
        if isinstance(other, ProbabilityVector):
            vals = self.values * other.values
            return ProbabilityVector(self.states, vals)
        # multiplication with Dataframe column
        # elif isinstance(other, pd.Series):
        #     vals = self.values * other.values.reshape(-1, 1)
        # # multiplication with numpy ndarray column
        # elif isinstance(other, np.ndarray):
        #     vals = self.values * other.reshape(-1, 1)
        # # element-wise matrix multiplication with Pandas matrix
        elif isinstance(other, pd.DataFrame):
            return self.df * other
        else:
            raise NotImplementedError


    # def __rmatmul__(self, other):
    #     # matrix multiplication with numpy ndarray matrix
    #     if isinstance(other, np.ndarray):
    #         return ProbabilityVector(dict(zip(self.states, other @ self.values)))
    #     # matrix multiplication with DataFrame matrix
    #     elif isinstance(other, pd.DataFrame):
    #         return ProbabilityVector(dict(zip(self.states, other.values @ self.values)))

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]


class ProbabilityMatrix:
    def __init__(self, states, obs, prob_vectors=None, sort=False):
        self.states = states
        self.obs = obs
        self.m = len(self.states)
        self.n = len(self.obs)

        if prob_vectors is not None:
            values = np.hstack([prob_vector.values for prob_vector in prob_vectors])
        else:
            values = np.zeros(shape=(self.m, self.n))

        self.df = pd.DataFrame(values, index=self.states, columns=self.obs)

    # @classmethod
    # def initialize(cls, states: list, observables: list):
    #     size = len(states)
    #     rand = np.random.rand(size, len(observables)) \
    #            / (size ** 2) + 1 / size
    #     rand /= rand.sum(axis=1).reshape(-1, 1)
    #     aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
    #     pvec = [ProbabilityVector(x) for x in aggr]
    #     return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, states, obs, values):
        prob_vectors = [ProbabilityVector(states, values[:, obs_i]) for obs_i in obs]
        return cls(states, obs, prob_vectors)

    @property
    def T(self):
        transposed_matrix = ProbabilityMatrix(self.states, self.obs)
        transposed_matrix.df = self.df.T
        return transposed_matrix

    @property
    def log(self):
        log_matrix = ProbabilityMatrix(self.states, self.obs)
        log_matrix.df = np.log(self.df)
        #log_matrix.df = np.log(self.df.mask(self.df <= 0)).fillna(0)
        return log_matrix

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def argmax_row(self):
        return self.df.idxmax(axis=1)

    @property
    def max_row(self):
        return ProbabilityVector(self.states, self.df.values.max(axis=1))

    def assign_dateframe(self, df):
        self.df = df

    def __repr__(self):
        return "Probability matrix states: {} -> values: {}.".format(
            self.states, self.df.values.shape)

    # row-wise product with Probability Vector
    def __mul__(self, other):
        if isinstance(other, ProbabilityVector):
            new_df = self.df * other.values.squeeze() #row-wise
            new = ProbabilityMatrix(self.states, self.obs)
            new.df = new_df.copy()
            return new
        else:
            raise NotImplementedError

    # row-wise sum with Probability Vector
    def __add__(self, other):
        if isinstance(other, ProbabilityVector):
            new_df = self.df + other.values.squeeze() #row-wise
            new = ProbabilityMatrix(self.states, self.obs)
            new.df = new_df.copy()
            return new
        else:
            raise NotImplementedError


    # matrix multiplication with Probability Vector or Probability matrix
    def __matmul__(self, other):
        if isinstance(other, ProbabilityVector):
            values = self.df.values @ other.values
            return ProbabilityVector(self.states, values)
        # matrix multiplication with Probability matrix
        # elif isinstance(other, ProbabilityMatrix):
        #     assert (self.df.shape[1] == other.df.shape[0], "Matrix multiplication isn't supported")
        #     values = self.df.values @ other.df.values
        #     return ProbabilityMatrix.from_numpy(self.states, self.obs, values)
        else:
            raise NotImplementedError

    def __getitem__(self, pos):
        # requesting number
        if len(pos) == 2 and pos[0] in self.obs:
            column, row = pos
            return self.df[column][row]
        # requesting column
        elif pos in self.obs:
            column = pos
            return ProbabilityVector(self.states, self.df[column].values, sort=False)
        else:
            raise NotImplementedError

    def __setitem__(self, pos, value):
        # setting number
        if len(pos) == 2 and pos[0] in self.obs:
            column, row = pos
            self.df[column][row] = value
        # setting column
        elif pos in self.obs:
            column = pos
            self.df[column] = value.df
        else:
            raise NotImplementedError

