__all__ = ['Engine']

class Engine(object):
    trim = False

    def __init__(self, ensemble, policy):
        self.ensemble = ensemble
        self.policy = policy

    def run(self, env, step, potential):
        """Run the simulation with the current potentials."""
        raise NotImplementedError()
