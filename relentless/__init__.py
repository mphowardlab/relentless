import sys
assert sys.version_info >= (3,4), "relentless requires Python 3.4"

from . import core
from .core import *
from . import environment
from . import optimize
from . import potential
from . import simulate
