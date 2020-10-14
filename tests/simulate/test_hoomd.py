"""Unit tests for relentless.simulate.hoomd."""
import os
import tempfile
import unittest

import gsd.hoomd
import hoomd
import numpy as np

import relentless

class test_HOOMD(unittest.TestCase):
    """Unit tests for relentless.HOOMD"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.Directory(self._tmp.name)

    #mock gsd file for testing
    def create_gsd(self):
        with gsd.hoomd.open(name=self.directory.file('test.gsd'), mode='wb') as f:
            s = gsd.hoomd.Snapshot()
            s.particles.N = 5
            s.particles.types = ['A','B']
            s.particles.typeid = [0,0,1,1,1]
            s.particles.position = np.random.random(size=(5,3))
            s.configuration.box = [2,2,2,0,0,0]
            f.append(s)
        return f

    def test_initialize(self):
        """Test running initialization simulation operations."""
        ens = relentless.Ensemble(T=1.0, V=relentless.Cube(L=2.0), N={'A':2,'B':3})
        pot = relentless.PairMatrix(types=ens.types)
        for pair in pot:
            pot[pair]['r'] = np.array([1.,2.,3.])
            pot[pair]['u'] = np.array([2.,4.,6.])
            pot[pair]['f'] = np.array([-2.,-2.,-2.])

        #InitializeFromFile
        f = self.create_gsd()
        op = relentless.simulate.hoomd.InitializeFromFile(filename=f.file.name)
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertIsInstance(sim[op].neighbor_list, hoomd.md.nlist.tree)
        self.assertIsInstance(sim[op].pair_potential, hoomd.md.pair.table)

        #InitializeRandomly
        op = relentless.simulate.hoomd.InitializeRandomly(seed=1)
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertIsInstance(sim[op].neighbor_list, hoomd.md.nlist.tree)
        self.assertIsInstance(sim[op].pair_potential, hoomd.md.pair.table)

    def test_minimization(self):
        """Test running energy minimization simulation operation."""
        ens = relentless.Ensemble(T=100.0, V=relentless.Cube(L=10.0), N={'A':2,'B':3})
        pot = relentless.PairMatrix(types=ens.types)
        for pair in pot:
            pot[pair]['r'] = np.array([1.,2.,3.])
            pot[pair]['u'] = np.array([2.,4.,6.])
            pot[pair]['f'] = np.array([-2.,-2.,-2.])

        #MinimizeEnergy
        op = [relentless.simulate.hoomd.InitializeRandomly(seed=1),
              relentless.simulate.hoomd.MinimizeEnergy(energy_tolerance=0.1,
                                                       force_tolerance=0.1,
                                                       max_iterations=1000,
                                                       dt=0.5)
             ]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

    def test_integrators(self):
        """Test adding and removing integrator operations."""
        #BrownianIntegrator
        pass
        #LangevinIntegrator

        #NPTIntegrator

        #NVTIntegrator

    def test_run(self):
        """Test run simulation operations."""
        #Run
        pass
        #RunUpTo

    def test_callback(self):
        """Test callback classes."""
        #ThermodynamicsCallback
        pass
        #RDFCallback

    def test_analyzer(self):
        """Test ensemble analyzer simulation operation."""
        #AddEnsembleAnalyzer
        pass

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
