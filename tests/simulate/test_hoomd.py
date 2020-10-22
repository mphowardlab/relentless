"""Unit tests for relentless.simulate.hoomd."""
import tempfile
import unittest

import gsd.hoomd
import hoomd
import numpy as np

import relentless
from ..potential.test_pair import LinPot

class test_HOOMD(unittest.TestCase):
    """Unit tests for relentless.HOOMD"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.Directory(self._tmp.name)

    #mock (NVT) ensemble and potential for testing
    def ens_pot(self):
        ens = relentless.Ensemble(T=2.0, V=relentless.Cube(L=10.0), N={'A':2,'B':3})

        # setup potentials
        pot = LinPot(ens.types,params=('m',))
        for pair in pot.coeff:
            pot.coeff[pair]['m'] = 2.0
        pots = relentless.simulate.Potentials()
        pots.pair.potentials.append(pot)
        pots.pair.rmax = 3.0
        pots.pair.num = 4

        return (ens,pots)

    #mock gsd file for testing
    def create_gsd(self):
        with gsd.hoomd.open(name=self.directory.file('test.gsd'), mode='wb') as f:
            s = gsd.hoomd.Snapshot()
            s.particles.N = 5
            s.particles.types = ['A','B']
            s.particles.typeid = [0,0,1,1,1]
            s.particles.position = np.random.uniform(low=-5.0,high=5.0,size=(5,3))
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
                                                       dt=0.01)
             ]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

    def test_integrators(self):
        """Test adding and removing integrator operations."""
        init = relentless.simulate.hoomd.InitializeRandomly(seed=1)
        h = relentless.simulate.hoomd.HOOMD(operations=init)

        #BrownianIntegrator
        ens,pot = self.ens_pot()
        brn = relentless.simulate.hoomd.AddBrownianIntegrator(dt=0.5,
                                                              friction=1.0,
                                                              seed=2)
        brn_r = relentless.simulate.hoomd.RemoveBrownianIntegrator(add_op=brn)
        h.operations = [init,brn]
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[brn].integrator.enabled)
        brn_r(sim)
        self.assertIsNone(sim[brn].integrator)

        #LangevinIntegrator
        ens,pot = self.ens_pot()
        lgv = relentless.simulate.hoomd.AddLangevinIntegrator(dt=0.5,
                                                              friction=1.0,
                                                              seed=2)
        lgv_r = relentless.simulate.hoomd.RemoveLangevinIntegrator(add_op=lgv)
        h.operations = [init,lgv]
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[lgv].integrator.enabled)
        lgv_r(sim)
        self.assertIsNone(sim[lgv].integrator)

        #NPTIntegrator
        ens_npt = relentless.Ensemble(T=100.0, P=5.5, N={'A':2,'B':3})
        _,pot = self.ens_pot()
        ens_npt.V = relentless.Cube(L=10.0)
        npt = relentless.simulate.hoomd.AddNPTIntegrator(dt=0.5,
                                                         tau_T=1.0,
                                                         tau_P=1.5)
        npt_r = relentless.simulate.hoomd.RemoveNPTIntegrator(add_op=npt)
        h.operations = [init,npt]
        sim = h.run(ensemble=ens_npt, potentials=pot, directory=self.directory)
        self.assertTrue(sim[npt].integrator.enabled)
        npt_r(sim)
        self.assertIsNone(sim[npt].integrator)

        #NVTIntegrator
        ens,pot = self.ens_pot()
        nvt = relentless.simulate.hoomd.AddNVTIntegrator(dt=0.5,
                                                         tau_T=1.0)
        nvt_r = relentless.simulate.hoomd.RemoveNVTIntegrator(add_op=nvt)
        h.operations = [init,nvt]
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[nvt].integrator.enabled)
        nvt_r(sim)
        self.assertIsNone(sim[nvt].integrator)

    def test_run(self):
        """Test run simulation operations."""
        init = relentless.simulate.hoomd.InitializeRandomly(seed=1)
        h = relentless.simulate.hoomd.HOOMD(operations=init)

        #Run
        ens,pot = self.ens_pot()
        run = relentless.simulate.hoomd.Run(steps=1000)
        h.operations = [init,run]
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

        #RunUpTo
        ens,pot = self.ens_pot()
        run = relentless.simulate.hoomd.RunUpTo(step=999)
        h.operations = [init,run]
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

    def test_analyzer(self):
        """Test ensemble analyzer simulation operation."""
        ens,pot = self.ens_pot()
        init = relentless.simulate.hoomd.InitializeRandomly(seed=1)
        analyzer = relentless.simulate.hoomd.AddEnsembleAnalyzer(check_thermo_every=5,
                                                                 check_rdf_every=5,
                                                                 rdf_dr=1.0)
        run = relentless.simulate.hoomd.Run(steps=500)
        nvt = relentless.simulate.hoomd.AddNVTIntegrator(dt=0.1,
                                                         tau_T=1.0)
        op = [init,nvt,analyzer,run]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        thermo = sim[analyzer].thermo_callback

        #extract ensemble
        ens_ = analyzer.extract_ensemble(sim)
        self.assertIsNotNone(ens_.T)
        self.assertNotEqual(ens_.T, 0)
        self.assertIsNotNone(ens_.P)
        self.assertNotEqual(ens_.P, 0)
        self.assertIsNotNone(ens_.V)
        self.assertNotEqual(ens_.V.volume, 0)
        for i,j in ens_.rdf:
            self.assertEqual(ens_.rdf[i,j].table.shape, (len(pot.pair.r)-1,2))
        self.assertEqual(thermo.num_samples, 100)

        #reset callback
        thermo.reset()
        self.assertEqual(thermo.num_samples, 0)
        self.assertIsNone(thermo.T)
        self.assertIsNone(thermo.P)
        self.assertIsNone(thermo.V)

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
