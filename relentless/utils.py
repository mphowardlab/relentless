import os

try:
    _basestring = basestring
except NameError:
    _basestring = str

class TemporaryWorkingDirectory(object):
    """ Temporary working directory.
    """
    def __init__(self, directory):
        self.current = os.getcwd()
        self.directory = directory

        # ensure directory exists
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def __enter__(self):
        os.chdir(self.directory)
        return self.directory

    def __exit__(self, exception_type, exception_value, traceback):
        os.chdir(self.current)

def isstr(s):
    return isinstance(s, _basestring)
