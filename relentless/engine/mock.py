from relentless.engine import Engine
from relentless.environment import Policy

class Mock(Engine):
    def __init__(self, ensemble, policy=Policy()):
        super().__init__(ensemble, policy)

    def run(self, env, step, potentials):
        pass
