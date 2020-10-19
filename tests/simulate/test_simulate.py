"""Unit tests for simulate module."""
import tempfile
import unittest

import numpy as np

import relentless

class test_Simulation(unittest.TestCase):
    """Unit tests for relentless.Simulation"""

    #mock functions for use as operations
    class CheckEnsemble(relentless.simulate.SimulationOperation):
        def __call__(self, sim):
            try:
                sim[self].value = sim.constant_ens
            except AttributeError:
                sim[self].value = False

    class CheckPotential(relentless.simulate.SimulationOperation):
        def __call__(self, sim):
            try:
                sim[self].value = sim.constant_pot
            except AttributeError:
                sim[self].value = False

    def test_init(self):
        """Test creation from data."""
        operations = [self.CheckEnsemble(), self.CheckPotential()]
        options = {'constant_ens':True, 'constant_pot':True}

        #no operations, no options
        d = relentless.simulate.Simulation()
        self.assertCountEqual(d.operations, [])
        self.assertDictEqual(d.options, {})

        #with operations, no options
        d = relentless.simulate.Simulation(operations)
        self.assertCountEqual(d.operations, operations)
        self.assertDictEqual(d.options, {})

        #no operations, with options
        d = relentless.simulate.Simulation(**options)
        self.assertCountEqual(d.operations, [])
        self.assertDictEqual(d.options, options)

        #with operations, with options
        d = relentless.simulate.Simulation(operations, **options)
        self.assertCountEqual(d.operations, operations)
        self.assertDictEqual(d.options, options)

    def test_run(self):
        """Test run method."""
        ens = relentless.Ensemble(T=1.0, P=4.0, N={'A':2})
        pot = relentless.PairMatrix(types=ens.types)
        for pair in pot:
            pot[pair]['r'] = np.array([1.,2.,3.])
            pot[pair]['u'] = np.array([2.,3.,4.])
        dirc = 'mock'
        sim = relentless.simulate.Simulation()
        operations = [self.CheckEnsemble(), self.CheckPotential()]
        options = {'constant_ens':True, 'constant_pot':True}

        #no operations, no options
        sim = relentless.simulate.Simulation()
        sim_ = sim.run(ensemble=ens, potentials=pot, directory=dirc)
        with self.assertRaises(AttributeError):
            sim_[operations[0]].value
        with self.assertRaises(AttributeError):
            sim_[operations[1]].value

        #with operations, no options
        sim = relentless.simulate.Simulation(operations=operations)
        sim_ = sim.run(ensemble=ens, potentials=pot, directory=dirc)
        self.assertFalse(sim_[operations[0]].value)
        self.assertFalse(sim_[operations[1]].value)

        #with operations, options
        sim = relentless.simulate.Simulation(operations=operations, **options)
        sim_ = sim.run(ensemble=ens, potentials=pot, directory=dirc)
        self.assertTrue(sim_[operations[0]].value)
        self.assertTrue(sim_[operations[1]].value)

        #invalid operation type
        sim = relentless.simulate.Simulation(operations='a')
        with self.assertRaises(TypeError):
            sim.run(ensemble=ens, potential=pot, directory=dirc)

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
