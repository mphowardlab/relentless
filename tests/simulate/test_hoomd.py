"""Unit tests for relentless.simulate.hoomd."""
import tempfile
import unittest

import numpy as np

import relentless

class test_HOOMD(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.Directory(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
