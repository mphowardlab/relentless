"""Unit tests for relentless.simulate.hoomd."""
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

    #mock (NVT) ensemble and potential for testing
    def ens_pot(self):
        ens = relentless.Ensemble(T=100.0, V=relentless.Cube(L=10.0), N={'A':2,'B':3})
        pot = relentless.PairMatrix(types=ens.types)
        for pair in pot:
            pot[pair]['r'] = np.array([1.,2.,3.])
            pot[pair]['u'] = np.array([2.,4.,6.])
            pot[pair]['f'] = np.array([-2.,-2.,-2.])
        return (ens,pot)

    #mock gsd file for testing
    def create_gsd(self):
        with gsd.hoomd.open(name=self.directory.file('test.gsd'), mode='wb') as f:
            s = gsd.hoomd.Snapshot()
            s.particles.N = 5
            s.particles.types = ['A','B']
            s.particles.typeid = [0,0,1,1,1]
            s.particles.position = np.random.random(size=(5,3))
            s.configuration.box = [10,10,10,0,0,0]
            f.append(s)
        return f

    def test_initialize(self):
        """Test running initialization simulation operations."""
        #InitializeFromFile
        ens,pot = self.ens_pot()
        f = self.create_gsd()
        op = relentless.simulate.hoomd.InitializeFromFile(filename=f.file.name)
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertIsInstance(sim[op].neighbor_list, hoomd.md.nlist.tree)
        self.assertIsInstance(sim[op].pair_potential, hoomd.md.pair.table)

        #InitializeRandomly
        ens,pot = self.ens_pot()
        op = relentless.simulate.hoomd.InitializeRandomly(seed=1)
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertIsInstance(sim[op].neighbor_list, hoomd.md.nlist.tree)
        self.assertIsInstance(sim[op].pair_potential, hoomd.md.pair.table)

    def test_minimization(self):
        """Test running energy minimization simulation operation."""
        #MinimizeEnergy
        ens,pot = self.ens_pot()
        op = [relentless.simulate.hoomd.InitializeRandomly(seed=1),
              relentless.simulate.hoomd.MinimizeEnergy(energy_tolerance=1e-7,
                                                       force_tolerance=1e-7,
                                                       max_iterations=1000,
                                                       dt=10)
             ]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

    def test_integrators(self):
        """Test adding and removing integrator operations."""
        init = relentless.simulate.hoomd.InitializeRandomly(seed=1)

        #BrownianIntegrator
        ens,pot = self.ens_pot()
        brn = relentless.simulate.hoomd.AddBrownianIntegrator(dt=0.5,
                                                              friction=1.0,
                                                              seed=2)
        brn_r = relentless.simulate.hoomd.RemoveBrownianIntegrator(add_op=brn)
        op = [init,brn]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[brn].integrator.enabled)
        brn_r(sim)
        self.assertFalse(sim[brn].integrator.enabled)

        #LangevinIntegrator
        ens,pot = self.ens_pot()
        lgv = relentless.simulate.hoomd.AddLangevinIntegrator(dt=0.5,
                                                              friction=1.0,
                                                              seed=2)
        lgv_r = relentless.simulate.hoomd.RemoveLangevinIntegrator(add_op=lgv)
        op = [init,lgv]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[lgv].integrator.enabled)
        lgv_r(sim)
        self.assertFalse(sim[lgv].integrator.enabled)

        #NPTIntegrator
        ens_npt = relentless.Ensemble(T=100.0, P=5.5, N={'A':2,'B':3})
        _,pot = self.ens_pot()
        ens_npt.V = relentless.Cube(L=10.0)
        npt = relentless.simulate.hoomd.AddNPTIntegrator(dt=0.5,
                                                         tau_T=1.0,
                                                         tau_P=1.5)
        npt_r = relentless.simulate.hoomd.RemoveNPTIntegrator(add_op=npt)
        op = [init,npt]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens_npt, potentials=pot, directory=self.directory)
        self.assertTrue(sim[npt].integrator.enabled)
        npt_r(sim)
        self.assertFalse(sim[npt].integrator.enabled)

        #NVTIntegrator
        ens,pot = self.ens_pot()
        nvt = relentless.simulate.hoomd.AddNVTIntegrator(dt=0.5,
                                                         tau_T=1.0)
        nvt_r = relentless.simulate.hoomd.RemoveNVTIntegrator(add_op=nvt)
        op = [init,nvt]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[nvt].integrator.enabled)
        nvt_r(sim)
        self.assertFalse(sim[nvt].integrator.enabled)

    def test_run(self):
        """Test run simulation operations."""
        init = relentless.simulate.hoomd.InitializeRandomly(seed=1)

        #Run
        ens,pot = self.ens_pot()
        run = relentless.simulate.hoomd.Run(steps=1000)
        op = [init,run]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

        #RunUpTo
        ens,pot = self.ens_pot()
        run = relentless.simulate.hoomd.RunUpTo(step=999)
        op = [init,run]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

    def test_analyzer(self):
        """Test ensemble analyzer simulation operation."""
        ens,pot = self.ens_pot()
        init = relentless.simulate.hoomd.InitializeRandomly(seed=1)
        emin = relentless.simulate.hoomd.MinimizeEnergy(energy_tolerance=1e-7,
                                                        force_tolerance=1e-7,
                                                        max_iterations=1000,
                                                        dt=10)
        analyzer = relentless.simulate.hoomd.AddEnsembleAnalyzer(check_thermo_every=5,
                                                                 check_rdf_every=5,
                                                                 rdf_dr=1.0)
        op = [init,emin,analyzer]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

        #thermo callback
        thermo = sim[analyzer].thermo_callback
        self.assertAlmostEqual(thermo.T, 100.0)
        self.assertAlmostEqual(thermo.P, 5.0)
        self.assertAlmostEqual(thermo.V.volume, 1000.0)
        self.assertAlmostEqual(thermo.num_samples, 100)

        #rdf callback
        

        #extract ensemble
        
        #reset thermo properties
        thermo.reset()
        self.assertAlmostEqual(thermo.T, 0)
        self.assertAlmostEqual(thermo.P, 0)
        self.assertAlmostEqual(thermo.V.volume, 0)

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
