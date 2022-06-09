"""Unit tests for relentless.simulate.lammps."""
import sys
import tempfile
import unittest

try:
    import lammps
except ImportError:
    pass
import numpy

import relentless
from ..potential.test_pair import LinPot

@unittest.skipIf(not relentless.simulate.lammps._lammps_found,
                 "Compatible LAMMPS not installed")
class test_LAMMPS(unittest.TestCase):
    """Unit tests for relentless.LAMMPS"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.data.Directory(self._tmp.name)

    # mock (NVT) ensemble and potential for testing
    def ens_pot(self):
        ens = relentless.ensemble.Ensemble(T=2.0, V=relentless.volume.Cube(L=10.0), N={'1':2,'2':3})
        ens.P = 2.5

        # setup potentials
        pot = LinPot(ens.types,params=('m',))
        for pair in pot.coeff:
            pot.coeff[pair]['m'] = 2.0
        pots = relentless.simulate.Potentials()
        pots.pair.potentials.append(pot)
        pots.pair.rmax = 10.0
        pots.pair.num = 11

        return (ens,pots)

    def create_file(self):
        file_ = self.directory.file('test.data')
        with open(file_,'w') as f:
            f.write(('LAMMPS test data\n'
                     '\n'
                     '5 atoms\n'
                     '2 atom types\n'
                     '\n'
                     '-5.0 5.0 xlo xhi\n'
                     '-5.0 5.0 ylo yhi\n'
                     '-5.0 5.0 zlo zhi\n'
                     '\n'
                     'Atoms\n'
                     '\n'
                     '1 1 -4.0 -4.0 -4.0\n'
                     '2 1 -2.0 -2.0 -2.0\n'
                     '3 2 0.0 0.0 0.0\n'
                     '4 2 2.0 2.0 2.0\n'
                     '5 2 4.0 4.0 4.0\n'
                     '\n'
                     'Masses\n'
                     '\n'
                     '1 0.3\n'
                     '2 0.1'))
        return file_

    def test_initialize(self):
        """Test running initialization simulation operations."""
        # InitializeFromFile
        ens,pot = self.ens_pot()
        file_ = self.create_file()
        op = relentless.simulate.lammps.InitializeFromFile(filename=file_)
        l = relentless.simulate.lammps.LAMMPS(operations=op, quiet=False)
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertIsInstance(sim.lammps, lammps.lammps)
        self.assertEqual(sim.lammps.get_natoms(), 5)

        # InitializeRandomly
        op = relentless.simulate.lammps.InitializeRandomly(seed=1)
        l = relentless.simulate.lammps.LAMMPS(operations=op, quiet=False)
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertIsInstance(sim.lammps, lammps.lammps)
        self.assertEqual(sim.lammps.get_natoms(), 5)

    def test_minimization(self):
        """Test running energy minimization simulation operation."""
        ens,pot = self.ens_pot()
        file_ = self.create_file()
        init = relentless.simulate.lammps.InitializeFromFile(filename=file_)

        # MinimizeEnergy
        op = [init,
              relentless.simulate.lammps.MinimizeEnergy(energy_tolerance=1e-7,
                                                        force_tolerance=1e-7,
                                                        max_iterations=1000,
                                                        options={'max_evaluations':10000})
             ]
        l = relentless.simulate.lammps.LAMMPS(operations=op, quiet=False)
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)

        # check default value of max_evaluations
        emin = relentless.simulate.lammps.MinimizeEnergy(energy_tolerance=1e-7,
                                                         force_tolerance=1e-7,
                                                         max_iterations=1000,
                                                         options={})
        self.assertEqual(emin.options['max_evaluations'], None)
        l = relentless.simulate.lammps.LAMMPS(operations=[init,emin], quiet=False)
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertEqual(emin.options['max_evaluations'], 100*emin.max_iterations)

    def test_integrators(self):
        """Test adding and removing integrator operations."""
        init = relentless.simulate.lammps.InitializeRandomly(seed=1)
        l = relentless.simulate.lammps.LAMMPS(operations=init, quiet=False)

        # LangevinIntegrator
        # float friction
        ens,pot = self.ens_pot()
        lgv = relentless.simulate.lammps.AddLangevinIntegrator(dt=0.5,
                                                               friction=1.5,
                                                               seed=2)
        lgv_r = relentless.simulate.lammps.RemoveLangevinIntegrator(add_op=lgv)
        l.operations = [init, lgv]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_langevin)))
        lgv_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_langevin)))

        # dictionary friction
        lgv = relentless.simulate.lammps.AddLangevinIntegrator(dt=0.5,
                                                               friction={'1':2.0,'2':5.0},
                                                               seed=2)
        lgv_r = relentless.simulate.lammps.RemoveLangevinIntegrator(add_op=lgv)
        l.operations = [init, lgv]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_langevin)))
        lgv_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_langevin)))

        # single-type friction
        ens_1 = relentless.ensemble.Ensemble(T=2.0, V=relentless.volume.Cube(L=10.0), N={'1':2})
        lgv = relentless.simulate.lammps.AddLangevinIntegrator(dt=0.5,
                                                               friction={'1':3.0},
                                                               seed=2)
        lgv_r = relentless.simulate.lammps.RemoveLangevinIntegrator(add_op=lgv)
        l.operations = [init, lgv]
        sim = l.run(ensemble=ens_1, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_langevin)))
        lgv_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_langevin)))

        # invalid-type friction
        lgv = relentless.simulate.lammps.AddLangevinIntegrator(dt=0.5,
                                                               friction={'2':5.0,'3':2.0},
                                                               seed=2)
        l.operations = [init, lgv]
        with self.assertRaises(KeyError):
            sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)

        # VerletIntegrator - NVE
        ens,pot = self.ens_pot()
        vrl = relentless.simulate.lammps.AddVerletIntegrator(dt=0.5)
        vrl_r = relentless.simulate.lammps.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [init, vrl]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        tb = relentless.simulate.BerendsenThermostat(T=1, tau=0.5)
        vrl = relentless.simulate.lammps.AddVerletIntegrator(dt=0.5, thermostat=tb)
        vrl_r = relentless.simulate.lammps.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [init, vrl]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        bb = relentless.simulate.BerendsenBarostat(P=1, tau=0.5)
        vrl = relentless.simulate.lammps.AddVerletIntegrator(dt=0.5, barostat=bb)
        vrl_r = relentless.simulate.lammps.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [init, vrl]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        vrl = relentless.simulate.lammps.AddVerletIntegrator(dt=0.5, thermostat=tb, barostat=bb)
        vrl_r = relentless.simulate.lammps.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [init, vrl]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        # VerletIntegrator - NVT
        tn = relentless.simulate.NoseHooverThermostat(T=1, tau=0.5)
        vrl = relentless.simulate.lammps.AddVerletIntegrator(dt=0.5, thermostat=tn)
        vrl_r = relentless.simulate.lammps.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [init, vrl]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        vrl = relentless.simulate.lammps.AddVerletIntegrator(dt=0.5, thermostat=tn, barostat=bb)
        vrl_r = relentless.simulate.lammps.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [init, vrl]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        # VerletIntegrator - NPH
        bm = relentless.simulate.MTKBarostat(P=1, tau=0.5)
        vrl = relentless.simulate.lammps.AddVerletIntegrator(dt=0.5, barostat=bm)
        vrl_r = relentless.simulate.lammps.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [init, vrl]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        vrl = relentless.simulate.lammps.AddVerletIntegrator(dt=0.5, thermostat=tb, barostat=bm)
        vrl_r = relentless.simulate.lammps.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [init, vrl]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        # VerletIntegrator - NPT
        vrl = relentless.simulate.lammps.AddVerletIntegrator(dt=0.5, thermostat=tn, barostat=bm)
        vrl_r = relentless.simulate.lammps.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [init, vrl]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        # VerletIntegrator - incorrect
        with self.assertRaises(TypeError):
            vrl = relentless.simulate.lammps.AddVerletIntegrator(dt=0.5, thermostat=bb, barostat=tb)
            l.operations = [init, vrl]
            sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)

    def test_run(self):
        """Test run simulation operations."""
        init = relentless.simulate.lammps.InitializeRandomly(seed=1)
        l = relentless.simulate.lammps.LAMMPS(operations=init, quiet=False)

        # Run
        ens,pot = self.ens_pot()
        run = relentless.simulate.lammps.Run(steps=1000)
        l.operations = [init,run]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)

        # RunUpTo
        run = relentless.simulate.lammps.RunUpTo(step=999)
        l.operations = [init,run]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)

    def test_analyzer(self):
        """Test ensemble analyzer simulation operation."""
        ens,pot = self.ens_pot()
        init = relentless.simulate.lammps.InitializeRandomly(seed=1)
        analyzer = relentless.simulate.lammps.AddEnsembleAnalyzer(check_thermo_every=5,
                                                                  check_rdf_every=5,
                                                                  rdf_dr=1.0)
        run = relentless.simulate.lammps.Run(steps=500)
        lgv = relentless.simulate.lammps.AddLangevinIntegrator(dt=0.1,
                                                               friction=1.0,
                                                               seed=1)
        op = [init,lgv,analyzer,run]
        h = relentless.simulate.lammps.LAMMPS(operations=op, quiet=False)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

        # extract ensemble
        ens_ = analyzer.extract_ensemble(sim)
        self.assertIsNotNone(ens_.T)
        self.assertNotEqual(ens_.T, 0)
        self.assertIsNotNone(ens_.P)
        self.assertNotEqual(ens_.P, 0)
        self.assertIsNotNone(ens_.V)
        self.assertNotEqual(ens_.V.volume, 0)
        for i,j in ens_.rdf:
            self.assertEqual(ens_.rdf[i,j].table.shape, (len(pot.pair.r)-1,2))

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
