from . import core

class Lonestar(core.Environment):
    mpiexec = 'ibrun'
    always_wrap = False

    def __init__(self, path, mock=False):
        super().__init__(path, mock)

class Stampede2(core.Environment):
    mpiexec = 'ibrun'
    always_wrap = False

    def __init__(self, path, mock=False):
        super().__init__(path, mock)
