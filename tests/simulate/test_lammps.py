"""Unit tests for relentless.simulate.lammps."""
import contextlib
import io
import sys
from parameterized import parameterized_class
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
@parameterized_class([{'dim': 2}, {'dim': 3}],
    class_name_func=lambda cls, num, params_dict: "{}_{}d".format(cls.__name__,params_dict['dim']))
class test_LAMMPS(unittest.TestCase):
    """Unit tests for relentless.LAMMPS"""

    def setUp(self):
        if relentless.mpi.world.rank_is_root:
            self._tmp = tempfile.TemporaryDirectory()
            directory = self._tmp.name
        else:
            directory = None
        directory = relentless.mpi.world.bcast(directory)
        self.directory = relentless.data.Directory(directory)

    # mock (NVT) ensemble and potential for testing
    def ens_pot(self):
        if self.dim == 3:
            ens = relentless.ensemble.Ensemble(T=2.0, V=relentless.extent.Cube(L=10.0), N={'1':2,'2':3})
        elif self.dim == 2:
            ens = relentless.ensemble.Ensemble(T=2.0, V=relentless.extent.Square(L=10.0), N={'1':2,'2':3})
        else:
            raise ValueError('LAMMPS supports 2d and 3d simulations')
        ens.P = 2.5
        # setup potentials
        pot = LinPot(ens.types,params=('m',))
        for pair in pot.coeff:
            pot.coeff[pair]['m'] = -2.0
        pots = relentless.simulate.Potentials()
        pots.pair.potentials.append(pot)
        pots.pair.rmax = 10.0
        pots.pair.num = 11

        return (ens,pots)

    def create_file(self):
        file_ = self.directory.file('test.data')
        
        if self.dim == 3:
            zlo = '-5.0'
            zhi = '5.0'
            z1 = '-4.0'
            z2 = '-2.0'
            z3 = '0.0'
            z4 = '2.0'
            z5 = '4.0'
        elif self.dim == 2: 
            zlo = '-0.1'
            zhi = '0.1'
            z1 = '0.0'
            z2 = '0.0'
            z3 = '0.0'
            z4 = '0.0'
            z5 = '0.0'
        else:
            raise ValueError('LAMMPS supports 2d and 3d simulations')
        
        with open(file_,'w') as f:
            f.write(('LAMMPS test data\n'
                    '\n'
                    '5 atoms\n'
                    '2 atom types\n'
                    '\n'
                    '-5.0 5.0 xlo xhi\n'
                    '-5.0 5.0 ylo yhi\n'
                    '{} {} zlo zhi\n'
                    '\n'
                    'Atoms\n'
                    '\n'
                    '1 1 -4.0 -4.0 {}\n'
                    '2 1 -2.0 -2.0 {}\n'
                    '3 2 0.0 0.0 {}\n'
                    '4 2 2.0 2.0 {}\n'
                    '5 2 4.0 4.0 {}\n'
                    '\n'
                    'Masses\n'
                    '\n'
                    '1 0.3\n'
                    '2 0.1').format(zlo, zhi, z1, z2, z3, z4, z5))
        return file_

    def test_initialize(self):
        """Test running initialization simulation operations."""
        # InitializeFromFile
        ens,pot = self.ens_pot()
        file_ = self.create_file()
        op = relentless.simulate.InitializeFromFile(filename=file_)
        op.lammps_types = {'1': 1, '2': 2}
        l = relentless.simulate.LAMMPS(op, dimension=self.dim)
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertIsInstance(sim.lammps, lammps.lammps)
        self.assertEqual(sim.lammps.get_natoms(), 5)

        #InitializeRandomly
        op = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        l = relentless.simulate.LAMMPS(op, dimension=self.dim)
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertIsInstance(sim.lammps, lammps.lammps)
        self.assertEqual(sim.lammps.get_natoms(), 5)

    def test_random_initialize_options(self):
        # no T
        ens,pot = self.ens_pot()
        op = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V)
        h = relentless.simulate.LAMMPS(op, dimension=self.dim)
        h.run(potentials=pot, directory=self.directory)

        # no T + mass
        m = {i: idx+1 for idx,i in enumerate(ens.N)}
        op = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, masses=m)
        h = relentless.simulate.LAMMPS(op, dimension=self.dim)
        h.run(potentials=pot, directory=self.directory)

        # T + mass
        op = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T, masses=m)
        h = relentless.simulate.LAMMPS(op, dimension=self.dim)
        h.run(potentials=pot, directory=self.directory)

    def test_minimization(self):
        """Test running energy minimization simulation operation."""
        ens,pot = self.ens_pot()
        file_ = self.create_file()
        init = relentless.simulate.InitializeFromFile(filename=file_)
        init.lammps_types = {'1': 1, '2': 2}

        #MinimizeEnergy
        emin = relentless.simulate.MinimizeEnergy(energy_tolerance=1e-7,
                                                  force_tolerance=1e-7,
                                                  max_iterations=1000,
                                                  options={'max_evaluations': 10000})
        l = relentless.simulate.LAMMPS(init, operations=[emin], dimension=self.dim)
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertEqual(emin.options['max_evaluations'], 10000)

        #check default value of max_evaluations
        emin = relentless.simulate.MinimizeEnergy(energy_tolerance=1e-7,
                                                  force_tolerance=1e-7,
                                                  max_iterations=1000,
                                                  options={})
        l = relentless.simulate.LAMMPS(init, operations=[emin], dimension=self.dim)
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertEqual(emin.options['max_evaluations'], 100*emin.max_iterations)

    def test_integrators(self):
        """Test adding and removing integrator operations."""
        ens,pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        l = relentless.simulate.LAMMPS(init, dimension=self.dim)

        # LangevinIntegrator
        # float friction
        lgv = relentless.simulate.AddLangevinIntegrator(dt=0.5,
                                                        T=ens.T,
                                                        friction=1.5,
                                                        seed=2)
        lgv_r = relentless.simulate.RemoveLangevinIntegrator(add_op=lgv)
        l.operations = [lgv]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_langevin)))
        lgv_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_langevin)))

        #dictionary friction
        lgv = relentless.simulate.AddLangevinIntegrator(dt=0.5,
                                                        T=ens.T,
                                                        friction={'1':2.0,'2':5.0},
                                                        seed=2)
        lgv_r = relentless.simulate.RemoveLangevinIntegrator(add_op=lgv)
        l.operations = [lgv]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_langevin)))
        lgv_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_langevin)))

        # single-type friction
        init_1 = relentless.simulate.InitializeRandomly(seed=1, N={'1':2}, V=relentless.extent.Cube(L=10.0), T=2.0)
        lgv = relentless.simulate.AddLangevinIntegrator(dt=0.5,
                                                        T=ens.T,
                                                        friction={'1':3.0},
                                                        seed=2)
        lgv_r = relentless.simulate.RemoveLangevinIntegrator(add_op=lgv)
        l.initializer = init_1
        l.operations = [lgv]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertTrue(sim.lammps.has_id('fix',str(lgv._fix_langevin)))
        lgv_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_nve)))
        self.assertFalse(sim.lammps.has_id('fix',str(lgv._fix_langevin)))

        #invalid-type friction
        lgv = relentless.simulate.AddLangevinIntegrator(dt=0.5,
                                                        T=ens.T,
                                                        friction={'2':5.0,'3':2.0},
                                                        seed=2)
        l.initializer = init
        l.operations = [lgv]
        with self.assertRaises(KeyError):
            sim = l.run(potentials=pot, directory=self.directory)

        # VerletIntegrator - NVE
        vrl = relentless.simulate.AddVerletIntegrator(dt=0.5)
        vrl_r = relentless.simulate.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [vrl]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        tb = relentless.simulate.BerendsenThermostat(T=1, tau=0.5)
        vrl = relentless.simulate.AddVerletIntegrator(dt=0.5, thermostat=tb)
        vrl_r = relentless.simulate.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [vrl]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        bb = relentless.simulate.BerendsenBarostat(P=1, tau=0.5)
        vrl = relentless.simulate.AddVerletIntegrator(dt=0.5, barostat=bb)
        vrl_r = relentless.simulate.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [vrl]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        vrl = relentless.simulate.AddVerletIntegrator(dt=0.5, thermostat=tb, barostat=bb)
        vrl_r = relentless.simulate.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [vrl]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        # VerletIntegrator - NVT
        tn = relentless.simulate.NoseHooverThermostat(T=1, tau=0.5)
        vrl = relentless.simulate.AddVerletIntegrator(dt=0.5, thermostat=tn)
        vrl_r = relentless.simulate.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [vrl]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        vrl = relentless.simulate.AddVerletIntegrator(dt=0.5, thermostat=tn, barostat=bb)
        vrl_r = relentless.simulate.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [vrl]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        # VerletIntegrator - NPH
        bm = relentless.simulate.MTKBarostat(P=1, tau=0.5)
        vrl = relentless.simulate.AddVerletIntegrator(dt=0.5, barostat=bm)
        vrl_r = relentless.simulate.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [vrl]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        vrl = relentless.simulate.AddVerletIntegrator(dt=0.5, thermostat=tb, barostat=bm)
        vrl_r = relentless.simulate.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [vrl]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        #VerletIntegrator - NPT
        vrl = relentless.simulate.AddVerletIntegrator(dt=0.5, thermostat=tn, barostat=bm)
        vrl_r = relentless.simulate.RemoveVerletIntegrator(add_op=vrl)
        l.operations = [vrl]
        sim = l.run(potentials=pot, directory=self.directory)
        self.assertTrue(sim.lammps.has_id('fix',str(vrl._fix)))
        vrl_r(sim)
        self.assertFalse(sim.lammps.has_id('fix',str(vrl._fix)))

        # VerletIntegrator - incorrect
        with self.assertRaises(TypeError):
            vrl = relentless.simulate.AddVerletIntegrator(dt=0.5, thermostat=bb, barostat=tb)
            l.operations = [vrl]
            sim = l.run(potentials=pot, directory=self.directory)

    def test_run(self):
        """Test run simulation operations."""
        ens,pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        l = relentless.simulate.LAMMPS(init, dimension=self.dim)

        # Run
        run = relentless.simulate.Run(steps=1000)
        l.operations = [run]
        sim = l.run(potentials=pot, directory=self.directory)

        #RunUpTo
        run = relentless.simulate.RunUpTo(step=999)
        l.operations = [run]
        sim = l.run(potentials=pot, directory=self.directory)

    def test_analyzer(self):
        """Test ensemble analyzer simulation operation."""
        ens,pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        analyzer = relentless.simulate.AddEnsembleAnalyzer(check_thermo_every=5,
                                                           check_rdf_every=5,
                                                           rdf_dr=1.0)
        run = relentless.simulate.Run(steps=500)
        lgv = relentless.simulate.AddLangevinIntegrator(dt=0.005,
                                                        T=ens.T,
                                                        friction=1.0,
                                                        seed=1)
        h = relentless.simulate.LAMMPS(init, operations=[lgv,analyzer,run], dimension=self.dim)
        sim = h.run(potentials=pot, directory=self.directory)

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
        if relentless.mpi.world.rank_is_root:
            self._tmp.cleanup()
            del self._tmp
        del self.directory

if __name__ == '__main__':
    unittest.main()
