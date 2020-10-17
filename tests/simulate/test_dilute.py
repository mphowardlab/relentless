"""Unit tests for dilute module."""
import tempfile
import unittest

import numpy as np

import relentless

class test_Dilute(unittest.TestCase):
    """Unit tests for relentless.simulate.Dilute"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.Directory(self._tmp.name)

    def test_run(self):
        """Test run method."""
        analyzer = relentless.simulate.dilute.AddEnsembleAnalyzer()
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

        d = relentless.simulate.Dilute(operations=analyzer)
        sim = d.run(ensemble=ens, potentials=pot, directory=self.directory)
        ens_ = analyzer.extract_ensemble(sim)
        self.assertAlmostEqual(ens_.P, 0.2197740)

        #defined force array
        ens = relentless.Ensemble(T=3.0, V=relentless.Cube(L=2.5), N={'A':2,'B':3})
        pot = relentless.PairMatrix(types=ens.types)
        for pair in pot:
            pot[pair]['r'] = np.array([1.,2.,3.])
            pot[pair]['u'] = np.array([3.,6.,9.])
            pot[pair]['f'] = np.array([-3.,-3.,-3.])
        nvt = relentless.simulate.dilute.AddNVTIntegrator()
        d = relentless.simulate.Dilute(operations=[nvt,analyzer], mock_option=True)
        sim = d.run(ensemble=ens, potentials=pot, directory=self.directory)
        ens_ = analyzer.extract_ensemble(sim)
        self.assertAlmostEqual(ens_.P, -0.2873865)

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
