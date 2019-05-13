import os

try:
    _basestring = basestring
except NameError:
    _basestring = str

def isstr(s):
    return isinstance(s, _basestring)
