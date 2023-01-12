"""Unit tests for relentless.simulate.lammps."""
import tempfile
import unittest

from parameterized import parameterized_class

try:
    import lammps
    import lammpsio
except ImportError:
    pass

import relentless
from tests.model.potential.test_pair import LinPot

_has_modules = (
    relentless.simulate.lammps._lammps_found
    and relentless.simulate.lammps._lammpsio_found
)


@unittest.skipIf(not _has_modules, "LAMMPS dependencies not installed")
@parameterized_class(
    [{"dim": 2}, {"dim": 3}],
    class_name_func=lambda cls, num, params_dict: "{}_{}d".format(
        cls.__name__, params_dict["dim"]
    ),
)
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
            ens = relentless.model.Ensemble(
                T=2.0, V=relentless.model.Cube(L=10.0), N={"1": 2, "2": 3}
            )
        elif self.dim == 2:
            ens = relentless.model.Ensemble(
                T=2.0, V=relentless.model.Square(L=10.0), N={"1": 2, "2": 3}
            )
        else:
            raise ValueError("LAMMPS supports 2d and 3d simulations")
        ens.P = 2.5
        # setup potentials
        pot = LinPot(ens.types, params=("m",))
        for pair in pot.coeff:
            pot.coeff[pair].update({"m": -2.0, "rmax": 1.0})
        pots = relentless.simulate.Potentials()
        pots.pair.potentials.append(pot)
        pots.pair.start = 1e-6
        pots.pair.stop = 2.0
        pots.pair.num = 3

        return (ens, pots)

    def create_file(self):
        file_ = self.directory.file("test.data")

        if relentless.mpi.world.rank_is_root:
            low = [-5, -5, -5 if self.dim == 3 else -0.1]
            high = [5, 5, 5 if self.dim == 3 else 0.1]
            snap = lammpsio.Snapshot(N=5, box=lammpsio.Box(low, high))
            snap.position[:, :2] = [[-4, -4], [-2, -2], [0, 0], [2, 2], [4, 4]]
            if self.dim == 3:
                snap.position[:, 2] = [-4, -2, 0, 2, 4]
            snap.typeid = [1, 1, 2, 2, 2]
            snap.mass = [0.3, 0.3, 0.1, 0.1, 0.1]
            lammpsio.DataFile.create(file_, snap)
        relentless.mpi.world.barrier()

        return file_

    def test_initialize(self):
        """Test running initialization simulation operations."""
        # InitializeFromFile
        ens, pot = self.ens_pot()
        file_ = self.create_file()
        op = relentless.simulate.InitializeFromFile(filename=file_)
        op.lammps_types = {"1": 1, "2": 2}
        lmp = relentless.simulate.LAMMPS(op, dimension=self.dim)
        sim = lmp.run(potentials=pot, directory=self.directory)
        self.assertIsInstance(sim.lammps, lammps.lammps)
        self.assertEqual(sim.lammps.get_natoms(), 5)

        # InitializeRandomly
        op = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        lmp = relentless.simulate.LAMMPS(op, dimension=self.dim)
        sim = lmp.run(potentials=pot, directory=self.directory)
        self.assertIsInstance(sim.lammps, lammps.lammps)
        self.assertEqual(sim.lammps.get_natoms(), 5)

    def test_random_initialize_options(self):
        # no T
        ens, pot = self.ens_pot()
        op = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V)
        h = relentless.simulate.LAMMPS(op, dimension=self.dim)
        h.run(potentials=pot, directory=self.directory)

        # T + diameters
        op = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, diameters={"1": 1.0, "2": 2.0}
        )
        h = relentless.simulate.LAMMPS(op, dimension=self.dim)
        h.run(potentials=pot, directory=self.directory)

        # no T + mass
        m = {i: idx + 1 for idx, i in enumerate(ens.N)}
        op = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, masses=m)
        h = relentless.simulate.LAMMPS(op, dimension=self.dim)
        h.run(potentials=pot, directory=self.directory)

        # T + mass
        op = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, T=ens.T, masses=m
        )
        h = relentless.simulate.LAMMPS(op, dimension=self.dim)
        h.run(potentials=pot, directory=self.directory)

    def test_minimization(self):
        """Test running energy minimization simulation operation."""
        ens, pot = self.ens_pot()
        file_ = self.create_file()
        init = relentless.simulate.InitializeFromFile(filename=file_)
        init.lammps_types = {"1": 1, "2": 2}

        # MinimizeEnergy
        emin = relentless.simulate.MinimizeEnergy(
            energy_tolerance=1e-7,
            force_tolerance=1e-7,
            max_iterations=1000,
            options={"max_evaluations": 10000},
        )
        lmp = relentless.simulate.LAMMPS(init, operations=[emin], dimension=self.dim)
        lmp.run(potentials=pot, directory=self.directory)
        self.assertEqual(emin.options["max_evaluations"], 10000)

        # check default value of max_evaluations
        emin = relentless.simulate.MinimizeEnergy(
            energy_tolerance=1e-7, force_tolerance=1e-7, max_iterations=1000, options={}
        )
        lmp = relentless.simulate.LAMMPS(init, operations=[emin], dimension=self.dim)
        lmp.run(potentials=pot, directory=self.directory)
        self.assertEqual(emin.options["max_evaluations"], 100 * emin.max_iterations)

    def test_langevin_dynamics(self):
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        lmp = relentless.simulate.LAMMPS(init, dimension=self.dim)

        # float friction
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=1, timestep=0.5, T=ens.T, friction=1.5, seed=2
        )
        lmp.operations = lgv
        lmp.run(pot, self.directory)

        # dictionary friction
        lgv.friction = {"1": 2.0, "2": 5.0}
        lmp.run(pot, self.directory)

        # single-type friction
        init_1 = relentless.simulate.InitializeRandomly(
            seed=1, N={"1": 2}, V=ens.V, T=ens.T
        )
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=1, timestep=0.5, T=ens.T, friction={"1": 3.0}, seed=2
        )
        lmp.initializer = init_1
        lmp.operations = lgv
        lmp.run(pot, self.directory)

        # invalid-type friction
        lgv.friction = {"2": 5.0, "3": 2.0}
        lmp.initializer = init
        lmp.operations = lgv
        with self.assertRaises(KeyError):
            lmp.run(pot, self.directory)

    def test_molecular_dynamics(self):
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        lmp = relentless.simulate.LAMMPS(init, dimension=self.dim)

        # VerletIntegrator - NVE
        vrl = relentless.simulate.RunMolecularDynamics(steps=1, timestep=1e-3)
        lmp.operations = vrl
        lmp.run(pot, self.directory)

        # NVT - Berendesen
        vrl.thermostat = relentless.simulate.BerendsenThermostat(T=1, tau=0.5)
        lmp.run(pot, self.directory)

        # NPT - Berendsen
        vrl.barostat = relentless.simulate.BerendsenBarostat(P=1, tau=0.5)
        lmp.run(pot, self.directory)

        # NPH - Berendsen
        vrl.thermostat = None
        lmp.run(pot, self.directory)

        # NVT - Nose Hoover
        vrl.thermostat = relentless.simulate.NoseHooverThermostat(T=1, tau=0.5)
        vrl.barostat = None
        lmp.run(pot, self.directory)

        # NPT - Nose Hoover + Berendsen
        vrl.barostat = relentless.simulate.BerendsenBarostat(P=1, tau=0.5)
        lmp.run(pot, self.directory)

        # NPT - Nose Hoover + MTK
        vrl.barostat = relentless.simulate.MTKBarostat(P=1, tau=0.5)
        lmp.run(pot, self.directory)

        # NPT - Berendsen + MTK
        vrl.thermostat = relentless.simulate.BerendsenThermostat(T=1, tau=0.5)
        lmp.run(pot, self.directory)

        # NPH - MTK
        vrl.thermostat = None
        lmp.run(pot, self.directory)

    def test_analyzer(self):
        """Test ensemble analyzer simulation operation."""
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, T=ens.T, diameters={"1": 1, "2": 1}
        )
        analyzer = relentless.simulate.EnsembleAverage(
            check_thermo_every=5, check_rdf_every=5, rdf_dr=1.0
        )
        analyzer2 = relentless.simulate.EnsembleAverage(
            check_thermo_every=10, check_rdf_every=10, rdf_dr=0.5
        )
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=500,
            timestep=0.001,
            T=ens.T,
            friction=1.0,
            seed=1,
            analyzers=[analyzer, analyzer2],
        )
        h = relentless.simulate.LAMMPS(init, operations=lgv, dimension=self.dim)
        sim = h.run(potentials=pot, directory=self.directory)

        # extract ensemble
        ens_ = sim[analyzer].ensemble
        self.assertIsNotNone(ens_.T)
        self.assertNotEqual(ens_.T, 0)
        self.assertIsNotNone(ens_.P)
        self.assertNotEqual(ens_.P, 0)
        self.assertIsNotNone(ens_.V)
        self.assertNotEqual(ens_.V.extent, 0)
        for i, j in ens_.rdf:
            # shape is determined by rmax for potential and rdf_dr
            self.assertEqual(ens_.rdf[i, j].table.shape, (2, 2))

        # extract ensemble from second analyzer, answers should be slightly different
        # for any quantities that fluctuate
        ens2_ = sim[analyzer2].ensemble
        self.assertIsNotNone(ens2_.T)
        self.assertNotEqual(ens2_.T, 0)
        self.assertNotEqual(ens2_.T, ens_.T)
        self.assertIsNotNone(ens2_.P)
        self.assertNotEqual(ens2_.P, 0)
        self.assertNotEqual(ens2_.P, ens_.P)
        self.assertIsNotNone(ens2_.V)
        self.assertNotEqual(ens2_.V.extent, 0)
        self.assertEqual(ens2_.V.extent, ens_.V.extent)
        for i, j in ens2_.rdf:
            self.assertEqual(ens2_.rdf[i, j].table.shape, (4, 2))

    def test_writetrajectory(self):
        """Test write trajectory simulation operation."""
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, T=ens.T, diameters={"1": 1, "2": 1}
        )
        analyzer = relentless.simulate.WriteTrajectory(
            filename="test_writetrajectory.lammpstrj",
            every=100,
            velocity=True,
            image=True,
            typeid=True,
            mass=True,
        )
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=500,
            timestep=0.001,
            T=ens.T,
            friction=1.0,
            seed=1,
            analyzers=[analyzer],
        )
        h = relentless.simulate.LAMMPS(init, operations=lgv, dimension=self.dim)
        sim = h.run(potentials=pot, directory=self.directory)

        # read trajectory file
        file = sim.directory.file("test_writetrajectory.lammpstrj")
        traj = lammpsio.DumpFile(
            filename=file,
            schema={
                "position": (0, 1, 2),
                "velocity": (3, 4, 5),
                "image": (6, 7, 8),
                "typeid": 9,
                "mass": 10,
            },
        )
        for snap in traj:
            self.assertEqual(snap.N, 5)
            self.assertIsNotNone(snap.velocity)
            self.assertIsNotNone(snap.image)
            self.assertCountEqual(snap.typeid, [1, 1, 2, 2, 2])
            self.assertCountEqual(snap.mass, [1, 1, 1, 1, 1])

    def tearDown(self):
        if relentless.mpi.world.rank_is_root:
            self._tmp.cleanup()
            del self._tmp
        del self.directory


if __name__ == "__main__":
    unittest.main()
