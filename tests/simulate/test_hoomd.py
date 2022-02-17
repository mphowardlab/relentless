"""Unit tests for relentless.simulate.hoomd."""
import tempfile
import unittest

try:
    import hoomd
except ImportError:
    pass
try:
    import gsd.hoomd
    _found_gsd = True
except ImportError:
    _found_gsd = False
import numpy

import relentless
from ..potential.test_pair import LinPot

_has_modules = (relentless.simulate.hoomd._hoomd_found and
                relentless.simulate.hoomd._freud_found and
                _found_gsd)

@unittest.skipIf(not _has_modules, "HOOMD, freud, and/or GSD not installed")
class test_HOOMD(unittest.TestCase):
    """Unit tests for relentless.HOOMD"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.data.Directory(self._tmp.name)

    #mock (NVT) ensemble and potential for testing
    def ens_pot(self):
        ens = relentless.ensemble.Ensemble(T=2.0, V=relentless.volume.Cube(L=20.0), N={'A':2,'B':3})

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
            s.particles.position = numpy.random.uniform(low=-5.0,high=5.0,size=(5,3))
            s.configuration.box = [20,20,20,0,0,0]
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
                                                       max_displacement=0.01)
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

        ens,pot = self.ens_pot()
        lgv = relentless.simulate.hoomd.AddLangevinIntegrator(dt=0.5,
                                                              friction={'A':1.5,'B':2.5},
                                                              seed=2)
        lgv_r = relentless.simulate.hoomd.RemoveLangevinIntegrator(add_op=lgv)
        h.operations = [init,lgv]
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[lgv].integrator.enabled)
        lgv_r(sim)
        self.assertIsNone(sim[lgv].integrator)

        #VerletIntegrator - NVE
        ens,pot = self.ens_pot()
        vrl = relentless.simulate.hoomd.AddVerletIntegrator(dt=0.5)
        vrl_r = relentless.simulate.hoomd.RemoveVerletIntegrator(add_op=vrl)
        h.operations = [init, vrl]
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[vrl].integrator.enabled)
        vrl_r(sim)
        self.assertIsNone(sim[vrl].integrator)

        #VerletIntegrator - NVE (Berendsen)
        tb = relentless.simulate.BerendsenThermostat(T=1, tau=0.5)
        vrl = relentless.simulate.hoomd.AddVerletIntegrator(dt=0.5, thermostat=tb)
        vrl_r = relentless.simulate.hoomd.RemoveVerletIntegrator(add_op=vrl)
        h.operations = [init, vrl]
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[vrl].integrator.enabled)
        vrl_r(sim)
        self.assertIsNone(sim[vrl].integrator)

        #VerletIntegrator - NVT
        tn = relentless.simulate.NoseHooverThermostat(T=1, tau=0.5)
        vrl = relentless.simulate.hoomd.AddVerletIntegrator(dt=0.5, thermostat=tn)
        vrl_r = relentless.simulate.hoomd.RemoveVerletIntegrator(add_op=vrl)
        h.operations = [init, vrl]
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[vrl].integrator.enabled)
        vrl_r(sim)
        self.assertIsNone(sim[vrl].integrator)

        #VerletIntegrator - NPH
        bm = relentless.simulate.MTKBarostat(P=1, tau=0.5)
        vrl = relentless.simulate.hoomd.AddVerletIntegrator(dt=0.5, barostat=bm)
        vrl_r = relentless.simulate.hoomd.RemoveVerletIntegrator(add_op=vrl)
        h.operations = [init, vrl]
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[vrl].integrator.enabled)
        vrl_r(sim)
        self.assertIsNone(sim[vrl].integrator)

        #VerletIntegrator - NPT
        vrl = relentless.simulate.hoomd.AddVerletIntegrator(dt=0.5, thermostat=tn, barostat=bm)
        vrl_r = relentless.simulate.hoomd.RemoveVerletIntegrator(add_op=vrl)
        h.operations = [init, vrl]
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim[vrl].integrator.enabled)
        vrl_r(sim)
        self.assertIsNone(sim[vrl].integrator)

        #VerletIntegrator - incorrect
        with self.assertRaises(TypeError):
            vrl = relentless.simulate.hoomd.AddVerletIntegrator(dt=0.5, thermostat=tb, barostat=bm)
            h.operations = [init, vrl]
            sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

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
        lgv = relentless.simulate.hoomd.AddLangevinIntegrator(dt=0.1,
                                                              friction=0.9,
                                                              seed=2)
        analyzer = relentless.simulate.hoomd.AddEnsembleAnalyzer(check_thermo_every=5,
                                                                 check_rdf_every=5,
                                                                 rdf_dr=1.0)
        run = relentless.simulate.hoomd.Run(steps=500)
        op = [init,lgv,analyzer,run]
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

    def test_self_interactions(self):
        """Test if self-interactions are excluded from rdf computation."""
        with gsd.hoomd.open(name=self.directory.file('mock.gsd'), mode='wb') as f:
            s = gsd.hoomd.Snapshot()
            s.particles.N = 4
            s.particles.types = ['A','B']
            s.particles.typeid = [0,1,0,1]
            s.particles.position = [[-1,-1,-1],[1,1,1],[1,-1,1],[-1,1,-1]]
            s.configuration.box = [8,8,8,0,0,0]
            f.append(s)

        ens = relentless.ensemble.Ensemble(T=2.0, V=relentless.volume.Cube(L=8.0), N={'A':2,'B':2})
        _,pot = self.ens_pot()
        init = relentless.simulate.hoomd.InitializeFromFile(filename=f.file.name)
        ig = relentless.simulate.hoomd.AddVerletIntegrator(dt=0.0)
        analyzer = relentless.simulate.hoomd.AddEnsembleAnalyzer(check_thermo_every=1,
                                                                 check_rdf_every=1,
                                                                 rdf_dr=0.1)
        run = relentless.simulate.hoomd.Run(steps=1)
        op = [init,ig,analyzer,run]
        h = relentless.simulate.hoomd.HOOMD(operations=op)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

        ens_ = analyzer.extract_ensemble(sim)
        for i,j in ens_.rdf:
            self.assertEqual(ens_.rdf[i,j].table[0,1], 0.0)

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
