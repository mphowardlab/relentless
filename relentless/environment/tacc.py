from . import core

class Lonestar(core.Environment):
    mpiexec = 'ibrun'
    always_wrap = False

class Stampede2(core.Environment):
    mpiexec = 'ibrun'
    always_wrap = False
