__all__ = ['Engine']

class Engine(object):
    trim = False

    def __init__(self, policy):
        self.policy = policy

    def run(self, env, step, ensemble, potential):
        """Run the simulation with the current potentials."""
        raise NotImplementedError()
