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

@unittest.skipIf(not relentless.simulate.lammps._lammps_found or
                 sys.version_info[:2] == (3,8),
                 "Compatible LAMMPS not installed")
class test_LAMMPS(unittest.TestCase):
    """Unit tests for relentless.LAMMPS"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = relentless.data.Directory(self._tmp.name)

    #mock (NVT) ensemble and potential for testing
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
        #InitializeFromFile
        ens,pot = self.ens_pot()
        file_ = self.create_file()
        op = relentless.simulate.lammps.InitializeFromFile(filename=file_)
        l = relentless.simulate.lammps.LAMMPS(operations=op, quiet=False)
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        pl = lammps.PyLammps(ptr=sim.lammps)
        self.assertIsNotNone(pl.system)

        #InitializeRandomly
        op = relentless.simulate.lammps.InitializeRandomly(seed=1)
        l = relentless.simulate.lammps.LAMMPS(operations=op, quiet=False)
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        pl = lammps.PyLammps(ptr=sim.lammps)
        self.assertIsNotNone(pl.system)

    def test_minimization(self):
        """Test running energy minimization simulation operation."""
        #MinimizeEnergy
        ens,pot = self.ens_pot()
        file_ = self.create_file()
        op = [relentless.simulate.lammps.InitializeFromFile(filename=file_),
              relentless.simulate.lammps.MinimizeEnergy(energy_tolerance=1e-7,
                                                        force_tolerance=1e-7,
                                                        max_iterations=1000,
                                                        dt=0.01)
             ]
        l = relentless.simulate.lammps.LAMMPS(operations=op, quiet=False)
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)

    def test_integrators(self):
        """Test adding and removing integrator operations."""
        default_fixes = [{'name':''}]

        init = relentless.simulate.lammps.InitializeRandomly(seed=1)
        l = relentless.simulate.lammps.LAMMPS(operations=init, quiet=False)

        #LangevinIntegrator
        #float friction
        ens,pot = self.ens_pot()
        lgv = relentless.simulate.lammps.AddLangevinIntegrator(dt=0.5,
                                                               friction=1.5,
                                                               seed=2)
        lgv_r = relentless.simulate.lammps.RemoveLangevinIntegrator(add_op=lgv)
        l.operations = [init, lgv]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        pl = lammps.PyLammps(ptr=sim.lammps)
        self.assertCountEqual(pl.fixes, default_fixes+[{'name':'1','style':'langevin','group':'all'},
                                                       {'name':'2','style':'nve','group':'all'}])
        lgv_r(sim)
        self.assertCountEqual(pl.fixes, default_fixes)

        #dictionary friction
        lgv = relentless.simulate.lammps.AddLangevinIntegrator(dt=0.5,
                                                               friction={'1':2.0,'2':5.0},
                                                               seed=2)
        lgv_r = relentless.simulate.lammps.RemoveLangevinIntegrator(add_op=lgv)
        l.operations = [init, lgv]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        pl = lammps.PyLammps(ptr=sim.lammps)
        self.assertCountEqual(pl.fixes, default_fixes+[{'name':'3','style':'langevin','group':'all'},
                                                       {'name':'4','style':'nve','group':'all'}])
        lgv_r(sim)
        self.assertCountEqual(pl.fixes, default_fixes)

        #single-type friction
        ens_1 = relentless.ensemble.Ensemble(T=2.0, V=relentless.volume.Cube(L=10.0), N={'1':2})
        lgv = relentless.simulate.lammps.AddLangevinIntegrator(dt=0.5,
                                                               friction={'1':3.0},
                                                               seed=2)
        lgv_r = relentless.simulate.lammps.RemoveLangevinIntegrator(add_op=lgv)
        l.operations = [init, lgv]
        sim = l.run(ensemble=ens_1, potentials=pot, directory=self.directory)
        pl = lammps.PyLammps(ptr=sim.lammps)
        self.assertCountEqual(pl.fixes, default_fixes+[{'name':'5','style':'langevin','group':'all'},
                                                       {'name':'6','style':'nve','group':'all'}])
        lgv_r(sim)
        self.assertCountEqual(pl.fixes, default_fixes)

        #invalid-type friction
        lgv = relentless.simulate.lammps.AddLangevinIntegrator(dt=0.5,
                                                               friction={'2':5.0,'3':2.0},
                                                               seed=2)
        l.operations = [init, lgv]
        with self.assertRaises(KeyError):
            sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)

        #NPTIntegrator
        ens_npt = relentless.ensemble.Ensemble(T=100.0, P=5.5, N={'A':2,'B':3})
        ens_npt.V = relentless.volume.Cube(L=10.0)
        npt = relentless.simulate.lammps.AddNPTIntegrator(dt=0.5,
                                                          tau_T=1.0,
                                                          tau_P=1.5)
        npt_r = relentless.simulate.lammps.RemoveNPTIntegrator(add_op=npt)
        l.operations = [init, npt]
        sim = l.run(ensemble=ens_npt, potentials=pot, directory=self.directory)
        pl = lammps.PyLammps(ptr=sim.lammps)
        self.assertCountEqual(pl.fixes, default_fixes+[{'name':'1','style':'npt','group':'all'}])
        npt_r(sim)
        self.assertCountEqual(pl.fixes, default_fixes)

        #NVTIntegrator
        nvt = relentless.simulate.lammps.AddNVTIntegrator(dt=0.5,
                                                          tau_T=1.0)
        nvt_r = relentless.simulate.lammps.RemoveNVTIntegrator(add_op=nvt)
        l.operations = [init, nvt]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)
        pl = lammps.PyLammps(ptr=sim.lammps)
        self.assertEqual(pl.fixes[1]['style'], 'nvt')
        self.assertEqual(pl.fixes[1]['group'], 'all')
        nvt_r(sim)
        self.assertCountEqual(pl.fixes, default_fixes)

    def test_run(self):
        """Test run simulation operations."""
        init = relentless.simulate.lammps.InitializeRandomly(seed=1)
        l = relentless.simulate.lammps.LAMMPS(operations=init, quiet=False)

        #Run
        ens,pot = self.ens_pot()
        run = relentless.simulate.lammps.Run(steps=1000)
        l.operations = [init,run]
        sim = l.run(ensemble=ens, potentials=pot, directory=self.directory)

        #RunUpTo
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
        nvt = relentless.simulate.lammps.AddNVTIntegrator(dt=0.1,
                                                          tau_T=1.0)
        op = [init,nvt,analyzer,run]
        h = relentless.simulate.lammps.LAMMPS(operations=op,quiet=False)
        sim = h.run(ensemble=ens, potentials=pot, directory=self.directory)

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

    def tearDown(self):
        self._tmp.cleanup()

if __name__ == '__main__':
    unittest.main()
