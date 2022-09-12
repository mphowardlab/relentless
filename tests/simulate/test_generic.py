"""Unit tests for generic module."""
import sys
import tempfile
import unittest

import numpy

import relentless
from ..potential.test_pair import LinPot

class test_Generic(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.data.Directory(self._tmp.name)
        self.ensemble = relentless.ensemble.Ensemble(T=1.0, V=relentless.extent.Cube(L=2.0), N={'A':2,'B':3})

        self.potentials = relentless.simulate.Potentials()
        pot = LinPot(self.ensemble.types,params=('m',))
        for pair in pot.coeff:
            pot.coeff[pair]['m'] = 2.0
        self.potentials.pair.potentials.append(pot)
        self.potentials.pair.rmax = 3.0
        self.potentials.pair.num = 4
        self.potentials.pair.neighbor_buffer = 0.4

        self.ops = [relentless.simulate.InitializeRandomly(seed=2),
                    relentless.simulate.AddLangevinIntegrator(dt=0.1, friction=0.8, seed=2)]

    def test_basic(self):
        # Dilute
        dilute = relentless.simulate.dilute.Dilute(self.ops)
        dilute.run(self.ensemble, self.potentials, self.directory)

    def test_hoomd(self):
        # HOOMD
        try:
            hoomd = relentless.simulate.hoomd.HOOMD(self.ops)
            hoomd.run(self.ensemble, self.potentials, self.directory)
        except ImportError:
            self.skipTest('HOOMD not installed')

    def test_lammps(self):
        # LAMMPS
        try:
            lammps = relentless.simulate.lammps.LAMMPS(self.ops)
            lammps.run(self.ensemble, self.potentials, self.directory)
        except ImportError:
            self.skipTest('LAMMPS not installed')

    def test_invalid(self):
        #Invalid backend
        with self.assertRaises(NotImplementedError):
            sim = relentless.simulate.Simulation(self.ops)
            sim.run(self.ensemble, self.potentials, self.directory)

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
