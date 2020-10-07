"""Unit tests for simulate module."""
import tempfile
import unittest

import numpy as np

import relentless

class test_Simulation(unittest.TestCase):
    """Unit tests for relentless.Simulation"""

    #mock function for use as operation
    class OperationA(relentless.simulate.SimulationOperation):
        def __call__(self, sim):
            try:
                a = sim.a
            except AttributeError:
                a = False
            if a:
                sim.ensemble.T += 3.0
            else:
                for pair in sim.potentials:
                    sim.potentials[pair]['u'] += 2.0

    def test_init(self):
        """Test creation from data."""
        #no operations, no options
        sim = relentless.simulate.Simulation()
        self.assertEqual(sim.operations, [])
        self.assertEqual(sim.options, {})

        #with operations, no options
        sim = relentless.simulate.Simulation(operations=self.OperationA())
        self.assertIsInstance(sim.operations[0], self.OperationA)
        self.assertEqual(sim.options, {})

        #with operations, options
        sim = relentless.simulate.Simulation(operations=[self.OperationA()], a=True)
        self.assertIsInstance(sim.operations[0], self.OperationA)
        self.assertDictEqual(sim.options, {'a':True})

    def test_run(self):
        """Test run method."""
        ens = relentless.Ensemble(T=1.0, P=4.0, N={'A':2})
        pot = relentless.PairMatrix(types=ens.types)
        for pair in pot:
            pot[pair]['r'] = np.array([1.,2.,3.])
            pot[pair]['u'] = np.array([2.,3.,4.])
        dirc = 'mock'

        #no operations, no options
        sim = relentless.simulate.Simulation()
        sim_ = sim.run(ensemble=ens, potentials=pot, directory=dirc)
        self.assertAlmostEqual(sim_.ensemble.T, 1.0)
        np.testing.assert_allclose(sim_.potentials['A','A']['u'], np.array([2.,3.,4.]))

        #with operations, no options
        sim = relentless.simulate.Simulation(operations=self.OperationA())
        sim_ = sim.run(ensemble=ens, potentials=pot, directory=dirc)
        self.assertAlmostEqual(sim_.ensemble.T, 1.0)
        np.testing.assert_allclose(sim_.potentials['A','A']['u'], np.array([4.,5.,6.]))

        #with operations, options
        sim = relentless.simulate.Simulation(operations=self.OperationA(), a=True)
        sim_ = sim.run(ensemble=ens, potentials=pot, directory=dirc)
        self.assertAlmostEqual(sim_.ensemble.T, 4.0)
        np.testing.assert_allclose(sim_.potentials['A','A']['u'], np.array([4.,5.,6.]))

        #invalid operation type
        sim = relentless.simulate.Simulation(operations='a')
        with self.assertRaises(TypeError):
            sim_ = sim.run(ensemble=ens, potential=pot, directory=dirc)

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
