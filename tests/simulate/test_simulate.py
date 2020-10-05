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

class test_Dilute(unittest.TestCase):
    """Unit tests for relentless.Dilute"""

    #mock functions for use as operations
    class UpdateEnsemble(relentless.simulate.SimulationOperation):
        def __call__(self, sim):
            try:
                const = sim.constant_ens
            except AttributeError:
                const = False

            if not const:
                sim.ensemble.T += 2.0
                sim.ensemble.V = relentless.Cube(L=4.0)
                for t in sim.ensemble.types:
                    sim.ensemble.N[t] += 1.0

    class UpdatePotential(relentless.simulate.SimulationOperation):
        def __call__(self, sim):
            try:
                const = sim.constant_pot
            except AttributeError:
                const = False

            if not const:
                for pair in sim.potentials:
                    sim.potentials[pair]['r'] *= 2.0
                    sim.potentials[pair]['u'] *= 3.0

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.Directory(self._tmp.name)

    def test_init(self):
        """Test creation from data."""
        operations = [self.UpdateEnsemble(), self.UpdatePotential()]
        options = {'constant_ens':True, 'constant_pot':True}

        #no operations, no options
        d = relentless.simulate.Dilute()
        self.assertCountEqual(d.operations, [])
        self.assertDictEqual(d.options, {})

        #with operations, no options
        d = relentless.simulate.Dilute(operations)
        self.assertCountEqual(d.operations, operations)
        self.assertDictEqual(d.options, {})

        #no operations, with options
        d = relentless.simulate.Dilute(**options)
        self.assertCountEqual(d.operations, [])
        self.assertDictEqual(d.options, options)

        #with operations, with options
        d = relentless.simulate.Dilute(operations, **options)
        self.assertCountEqual(d.operations, operations)
        self.assertDictEqual(d.options, options)

    def test_run(self):
        """Test run method."""
        analyzer = relentless.simulate.dilute.AddEnsembleAnalyzer()
        operations = [self.UpdateEnsemble(),self.UpdatePotential()]
        ens = relentless.Ensemble(T=1.0, V=relentless.Cube(L=2.0), N={'A':2,'B':3})
        pot = relentless.PairMatrix(types=ens.types)
        for pair in pot:
            pot[pair]['r'] = np.array([1.,2.,3.])
            pot[pair]['u'] = np.array([2.,4.,6.])

        #invalid ensemble (non-NVT)
        ens_ = relentless.Ensemble(T=1, V=relentless.Cube(1), N={'A':2}, mu={'B':0.2})
        d = relentless.simulate.Dilute(analyzer)
        with self.assertRaises(ValueError):
            d.run(ensemble=ens_, potentials=pot, directory=self.directory)

        #invalid potential (r and u not defined)
        pot_ = relentless.PairMatrix(types=ens.types)
        with self.assertRaises(ValueError):
            d.run(ensemble=ens, potentials=pot_, directory=self.directory)

        #different run configurations
        #no operations (other than analyzer)
        d = relentless.simulate.Dilute(analyzer)
        sim = d.run(ensemble=ens, potentials=pot, directory=self.directory)
        ens_ = analyzer.extract_ensemble(sim)
        self.assertAlmostEqual(ens_.T, 1.0)
        self.assertAlmostEqual(ens_.V.volume, 8.0)
        self.assertAlmostEqual(ens_.P, 0.2197740)
        self.assertDictEqual(ens_.N.todict(), {'A':2,'B':3})
        for pair in ens_.rdf:
            np.testing.assert_allclose(ens_.rdf[pair].table,
                                       np.array([[1.,np.exp(-2./1.)],
                                                 [2.,np.exp(-4./1.)],
                                                 [3.,np.exp(-6./1.)]]))

        #with operations, no options
        d = relentless.simulate.Dilute(operations + [analyzer])
        sim = d.run(ensemble=ens, potentials=pot, directory=self.directory)
        ens_ = analyzer.extract_ensemble(sim)
        self.assertAlmostEqual(ens_.T, 3.0)
        self.assertAlmostEqual(ens_.V.volume, 64.0)
        self.assertAlmostEqual(ens_.P, 0.0302839)
        self.assertDictEqual(ens_.N.todict(), {'A':3,'B':4})
        for pair in ens_.rdf:
            np.testing.assert_allclose(ens_.rdf[pair].table,
                                       np.array([[2.,np.exp(-6./3.)],
                                                 [4.,np.exp(-12./3.)],
                                                 [6.,np.exp(-18./3.)]]))

        #defined force array
        for pair in pot:
            pot[pair]['f'] = np.array([-3.,-3.,-3.])
        #with operations, only one option enabled
        d = relentless.simulate.Dilute(operations + [analyzer], constant_pot=True)
        sim = d.run(ensemble=ens, potentials=pot, directory=self.directory)
        ens_ = analyzer.extract_ensemble(sim)
        self.assertAlmostEqual(ens_.T, 5.0)
        self.assertAlmostEqual(ens_.V.volume, 64.0)
        self.assertAlmostEqual(ens_.P, -1.7724031)
        self.assertDictEqual(ens_.N.todict(), {'A':4,'B':5})
        for pair in ens_.rdf:
            np.testing.assert_allclose(ens_.rdf[pair].table,
                                       np.array([[2.,np.exp(-6./5.)],
                                                 [4.,np.exp(-12./5.)],
                                                 [6.,np.exp(-18./5.)]]))

        #with operations, both options enabled
        d = relentless.simulate.Dilute(operations + [analyzer], constant_ens=True, constant_pot=True)
        sim = d.run(ensemble=ens, potentials=pot, directory=self.directory)
        ens_ = analyzer.extract_ensemble(sim)
        self.assertAlmostEqual(ens_.T, 5.0)
        self.assertAlmostEqual(ens_.V.volume, 64.0)
        self.assertAlmostEqual(ens_.P, -1.7724031)
        self.assertDictEqual(ens_.N.todict(), {'A':4,'B':5})
        for pair in ens_.rdf:
            np.testing.assert_allclose(ens_.rdf[pair].table,
                                       np.array([[2.,np.exp(-6./5.)],
                                                 [4.,np.exp(-12./5.)],
                                                 [6.,np.exp(-18./5.)]]))

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
