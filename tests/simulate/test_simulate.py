"""Unit tests for simulate module."""
import tempfile
import unittest

import numpy as np

import relentless

class test_SimulationInstance(unittest.TestCase):
    """Unit tests for relentless.SimulationInstance"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.Directory(self._tmp.name)

    def test_init(self):
        """Test creation from data."""
        options = {'constant_ens':True, 'constant_pot':False}
        ens = relentless.Ensemble(T=1.0, V=relentless.Cube(L=2.0), N={'A':2,'B':3})
        pot = relentless.PairMatrix(types=ens.types)
        for pair in pot:
            pot[pair]['r'] = np.array([1.,2.,3.])
            pot[pair]['u'] = np.array([2.,4.,6.])

        #no options
        sim = relentless.simulate.SimulationInstance(ens, pot, self.directory)
        self.assertEqual(sim.ensemble, ens)
        self.assertEqual(sim.potentials, pot)
        with self.assertRaises(AttributeError):
            sim.constant_ens

        #with options
        sim = relentless.simulate.SimulationInstance(ens, pot, self.directory, **options)
        self.assertEqual(sim.ensemble, ens)
        self.assertEqual(sim.potentials, pot)
        self.assertTrue(sim.constant_ens)
        self.assertFalse(sim.constant_pot)

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
