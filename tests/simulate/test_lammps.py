"""Unit tests for relentless.simulate.lammps."""
import tempfile
import unittest

import lammps
import numpy as np

import relentless
from ..potential.test_pair import LinPot

class test_LAMMPS(unittest.TestCase):
    """Unit tests for relentless.LAMMPS"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.Directory(self._tmp.name)

    #mock (NVT) ensemble and potential for testing
    def ens_pot(self):
        ens = relentless.Ensemble(T=2.0, V=relentless.Cube(L=10.0), N={'1':2,'2':3})

        # setup potentials
        pot = LinPot(ens.types,params=('m',))
        for pair in pot.coeff:
            pot.coeff[pair]['m'] = 2.0
        pots = relentless.simulate.Potentials()
        pots.pair.potentials.append(pot)
        pots.pair.rmax = 3.0
        pots.pair.num = 4

        return (ens,pots)

    def test_initialize(self):
        """Test running initialization simulation operations."""
        #InitializeRandomly
        ens,pot = self.ens_pot()
        op = relentless.simulate.lammps.InitializeRandomly(neighbor_buffer=0.4, seed=1)
        l = relentless.simulate.lammps.LAMMPS(operations=op)
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
