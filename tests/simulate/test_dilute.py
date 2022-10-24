"""Unit tests for dilute module."""
import tempfile
import unittest

import relentless

from ..potential.test_pair import LinPot

class test_Dilute(unittest.TestCase):
    """Unit tests for relentless.simulate.Dilute"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.data.Directory(self._tmp.name)

    def test_run(self):
        """Test run method."""
        init = relentless.simulate.InitializeRandomly(
                seed=42,
                N={'A':2,'B':3},
                V=relentless.extent.Cube(L=2.0),
                T=1.0)
        analyzer = relentless.simulate.AddEnsembleAnalyzer(
            check_thermo_every=1, check_rdf_every=1, rdf_dr=0.1)

        # set up potentials
        pot = LinPot(('A','B'),params=('m',))
        for pair in pot.coeff:
            pot.coeff[pair]['m'] = 2.0
        pots = relentless.simulate.Potentials()
        pots.pair.potentials.append(pot)
        pots.pair.rmax = 3.0
        pots.pair.num = 4

        d = relentless.simulate.Dilute(init, operations=[analyzer])
        sim = d.run(potentials=pots, directory=self.directory)
        ens_ = analyzer.extract_ensemble(sim)
        self.assertAlmostEqual(ens_.P, -207.5228556)

    def test_inf_potential(self):
        """Test potential with infinite value."""
        init = relentless.simulate.InitializeRandomly(
                seed=42,
                N={'A':2,'B':3},
                V=relentless.extent.Cube(L=2.0),
                T=1.0)
        analyzer = relentless.simulate.AddEnsembleAnalyzer(
            check_thermo_every=1, check_rdf_every=1, rdf_dr=0.1)

        # test with potential that has infinite potential at low r
        pot = relentless.potential.LennardJones(types=('A','B'))
        for pair in pot.coeff:
            pot.coeff[pair].update({'epsilon':1.0, 'sigma':1.0, 'rmax':3.0, 'shift':True})
        pots = relentless.simulate.Potentials()
        pots.pair.potentials.append(pot)
        pots.pair.rmax = 3.0
        pots.pair.num = 100

        d = relentless.simulate.Dilute(init, operations=[analyzer])
        warned = False
        try:
            sim = d.run(potentials=pots, directory=self.directory)
        except RuntimeWarning:
            warned = True
        self.assertFalse(warned)   # no warning should be raised
        ens_ = analyzer.extract_ensemble(sim)
        self.assertAlmostEqual(ens_.P, -1.1987890)

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
