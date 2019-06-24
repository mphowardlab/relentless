from relentless.engine import Engine
from relentless.environment import Policy

class Mock(Engine):
    def __init__(self, policy=Policy()):
        super().__init__(policy)

    def run(self, env, step, ensemble, potentials):
        pass
