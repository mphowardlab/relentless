"""Unit tests for simulate module."""
import tempfile
import unittest

import numpy as np

import relentless

class test_Default(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.Directory(self._tmp.name)

        self.ensemble = relentless.Ensemble(T=1.0, V=relentless.Cube(L=2.0), N={'A':2,'B':3})
        self.potentials = relentless.PairMatrix(types=self.ensemble.types)
        for pair in self.potentials:
            self.potentials[pair]['r'] = np.array([1.,2.,3.])
            self.potentials[pair]['u'] = np.array([2.,4.,6.])

    def test_basic(self):
        op = relentless.simulate.InitializeRandomly()
        dilute = relentless.simulate.Dilute([op])
        dilute.run(self.ensemble, self.potentials, self.directory)

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
