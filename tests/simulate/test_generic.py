"""Unit tests for generic module."""
import tempfile
import unittest

import numpy as np

import relentless
from ..potential.test_pair import LinPot

class test_Generic(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.data.Directory(self._tmp.name)
        self.ensemble = relentless.ensemble.Ensemble(T=1.0, V=relentless.volume.Cube(L=2.0), N={'A':2,'B':3})

        self.potentials = relentless.simulate.Potentials()
        pot = LinPot(self.ensemble.types,params=('m',))
        for pair in pot.coeff:
            pot.coeff[pair]['m'] = 2.0
        self.potentials.pair.potentials.append(pot)
        self.potentials.pair.rmax = 3.0
        self.potentials.pair.num = 4

    def test_basic(self):
        """Test valid and invalid operation calls."""
        ops = [relentless.simulate.InitializeRandomly(seed=2,neighbor_buffer=0.4),
               relentless.simulate.AddBrownianIntegrator(dt=0.1,friction=1,seed=1)]

        #Dilute
        dilute = relentless.simulate.dilute.Dilute(ops)
        dilute.run(self.ensemble, self.potentials, self.directory)

        #HOOMD
        try:
            hoomd = relentless.simulate.hoomd.HOOMD(ops)
            hoomd.run(self.ensemble, self.potentials, self.directory)
        except ImportError:
            pass

        #LAMMPS
        try:
            lammps = relentless.simulate.lammps.LAMMPS(ops)
            lammps.run(self.ensemble, self.potentials, self.directory)
        except ImportError:
            pass

        #Invalid backend
        with self.assertRaises(TypeError):
            sim = relentless.simulate.Simulation(ops)
            sim.run(self.ensemble, self.potentials, self.directory)

        #Invalid operation (in valid backend)
        try:
            ops = [relentless.simulate.InitializeRandomly(seed=1,neighbor_buffer=0.4),
                   relentless.simulate.AddBrownianIntegrator(dt=0.1,friction=0.5,seed=2)]
            lammps = relentless.simulate.lammps.LAMMPS([ops])
            with self.assertRaises(TypeError):
                lammps.run(self.ensemble, self.potentials, self.directory)
        except ImportError:
            pass

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
