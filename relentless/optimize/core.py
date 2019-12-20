__all__ = ['Optimizer','OptimizationProblem']

import numpy as np
import scipy.integrate

class Optimizer:
    def __init__(self, problem):
        self.problem = problem
        self.step = 0

    def run(self):
        raise NotImplementedError()

    def restart(self, env, step):
        """ Restart a calculation from a given step."""
        self.problem.restart(env, step)
        self.step = step

class OptimizationProblem:
    def __init__(self, engine):
        self.engine = engine
        self.step = 0

    def grad(self, env, step):
        raise NotImplementedError()

    def error(self, env, step):
        raise NotImplementedError()

    def restart(self, env, step):
        """ Restart a calculation from a given step."""
        with env.data(step):
            for pot in self.engine.potentials:
                pot.load(step)

    def get_variables(self):
        # flatten down variables for all potentials
        variables = []
        keys = set()
        for pot in self.engine.potentials:
            for key in pot.variables:
                if pot.variables[key] is not None:
                    for var in pot.variables[key]:
                        variables.append((pot,key,var))
                        keys.add(key)

        return variables,keys
