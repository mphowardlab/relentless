"""Unit tests for relentless.simulate.hoomd."""

import os
import pathlib
import tempfile
import unittest

import gsd
import gsd.hoomd
import lammpsio
import numpy
import scipy.stats
from packaging import version
from parameterized import parameterized_class

import relentless
from tests.model.potential.test_pair import LinPot

_has_modules = relentless.simulate.hoomd._hoomd_found

# silence warnings about Snapshot being deprecated
try:
    gsd_version = gsd.version.version
except AttributeError:
    gsd_version = gsd.__version__
if version.Version(gsd_version) >= version.Version("2.8.0"):
    HOOMDFrame = gsd.hoomd.Frame
    gsd_write_mode = "w"
else:
    HOOMDFrame = gsd.hoomd.Snapshot
    gsd_write_mode = "wb"


@unittest.skipIf(not _has_modules, "HOOMD dependencies not installed")
@parameterized_class(
    [{"dim": 2}, {"dim": 3}],
    class_name_func=lambda cls, num, params_dict: "{}_{}d".format(
        cls.__name__, params_dict["dim"]
    ),
)
class test_HOOMD(unittest.TestCase):
    """Unit tests for relentless.HOOMD"""

    def setUp(self):
        if relentless.mpi.world.rank_is_root:
            self._tmp = tempfile.TemporaryDirectory()
            directory = self._tmp.name
        else:
            directory = None
        directory = relentless.mpi.world.bcast(directory)
        self.directory = relentless.data.Directory(
            directory, create=relentless.mpi.world.rank_is_root
        )
        relentless.mpi.world.barrier()

    # mock (NVT) ensemble and potential for testing
    def ens_pot(self):
        if self.dim == 3:
            ens = relentless.model.Ensemble(
                T=2.0, V=relentless.model.Cube(L=20.0), N={"A": 2, "B": 3}
            )
        elif self.dim == 2:
            ens = relentless.model.Ensemble(
                T=2.0, V=relentless.model.Square(L=20.0), N={"A": 2, "B": 3}
            )
        else:
            raise ValueError("HOOMD supports 2d and 3d simulations")

        # setup potentials
        pot = LinPot(ens.types, params=("m",))
        for pair in pot.coeff:
            pot.coeff[pair].update({"m": -2.0, "rmax": 1.0})
        pots = relentless.simulate.Potentials()
        pots.pair = relentless.simulate.PairPotentialTabulator(
            pot, start=0.0, stop=2.0, num=3, neighbor_buffer=0.4
        )

        return (ens, pots)

    # mock gsd file for testing
    def create_gsd_file(self):
        filename = self.directory.file("test.gsd")
        if relentless.mpi.world.rank_is_root:
            with gsd.hoomd.open(name=filename, mode=gsd_write_mode) as f:
                s = HOOMDFrame()
                s.particles.N = 5
                s.particles.types = ["A", "B"]
                s.particles.typeid = [0, 0, 1, 1, 1]
                if self.dim == 3:
                    s.particles.position = numpy.random.uniform(
                        low=-5.0, high=5.0, size=(5, 3)
                    )
                    s.configuration.box = [20, 20, 20, 0, 0, 0]
                elif self.dim == 2:
                    s.particles.position = numpy.random.uniform(
                        low=-5.0, high=5.0, size=(5, 3)
                    )
                    s.particles.position[:, 2] = 0
                    s.configuration.box = [20, 20, 0, 0, 0, 0]
                else:
                    raise ValueError("HOOMD supports 2d and 3d simulations")
                f.append(s)
        relentless.mpi.world.barrier()
        return filename

    def create_lammps_file(self):
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

    def test_initialize_from_gsd_file(self):
        ens, pot = self.ens_pot()
        f = self.create_gsd_file()
        op = relentless.simulate.InitializeFromFile(filename=f)
        h = relentless.simulate.HOOMD(op)
        h.run(pot, self.directory)

        # Run in a different directory
        with self.directory:
            d = self.directory.directory(
                "run", create=relentless.mpi.world.rank_is_root
            )
            relentless.mpi.world.barrier()
            op.filename = pathlib.Path(f).name
            h.initializer = relentless.simulate.InitializeFromFile(pathlib.Path(f).name)
            h.run(pot, d)

    def test_initialize_from_lammps_file(self):
        """Test running initialization simulation operations."""
        pot = LinPot(("1", "2"), params=("m",))
        for pair in pot.coeff:
            pot.coeff[pair].update({"m": -2.0, "rmax": 1.0})
        pots = relentless.simulate.Potentials()
        pots.pair = relentless.simulate.PairPotentialTabulator(
            pot, start=1e-6, stop=2.0, num=10, neighbor_buffer=0.1
        )

        f = self.create_lammps_file()
        op = relentless.simulate.InitializeFromFile(filename=f)
        h = relentless.simulate.HOOMD(op)
        h.run(pots, self.directory)

    def test_initialize_randomly(self):
        ens, pot = self.ens_pot()
        op = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        h = relentless.simulate.HOOMD(op)
        h.run(pot, self.directory)

        # no T
        ens, pot = self.ens_pot()
        h.initializer = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V)
        h.run(pot, self.directory)

        # T + diameters
        h.initializer = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, diameters={"A": 1.0, "B": 2.0}
        )
        h.run(pot, self.directory)

        # no T + mass
        m = {i: idx + 1 for idx, i in enumerate(ens.N)}
        h.initializer = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, masses=m
        )
        h.run(pot, self.directory)

        # T + mass
        h.initializer = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, T=ens.T, masses=m
        )
        h.run(pot, self.directory)

    def test_device(self):
        ens, pot = self.ens_pot()
        op = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)

        # auto
        h = relentless.simulate.HOOMD(op, device="auto")
        h.run(pot, self.directory)

        # cpu
        h = relentless.simulate.HOOMD(op, device="cpu")
        h.run(pot, self.directory)
        h = relentless.simulate.HOOMD(op, device="CPU")
        h.run(pot, self.directory)

        # can't do gpu on systems that don't have one available, so won't test
        # can check for a disallowed value though

        h = relentless.simulate.HOOMD(op, device="knl")
        with self.assertRaises(ValueError):
            h.run(pot, self.directory)

    def test_minimization(self):
        """Test running energy minimization simulation operation."""
        # MinimizeEnergy
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)

        emin = relentless.simulate.MinimizeEnergy(
            energy_tolerance=1e-7,
            force_tolerance=1e-7,
            max_iterations=1000,
            options={"max_displacement": 0.5, "steps_per_iteration": 50},
        )
        h = relentless.simulate.HOOMD(init, emin)
        h.run(pot, self.directory)
        self.assertEqual(emin.options["max_displacement"], 0.5)
        self.assertEqual(emin.options["steps_per_iteration"], 50)

        # error check for missing max_displacement
        with self.assertRaises(KeyError):
            emin = relentless.simulate.MinimizeEnergy(
                energy_tolerance=1e-7,
                force_tolerance=1e-7,
                max_iterations=1000,
            )
            h = relentless.simulate.HOOMD(init, emin)
            h.run(pot, self.directory)

        # check default value for max_evaluations
        emin = relentless.simulate.MinimizeEnergy(
            energy_tolerance=1e-7,
            force_tolerance=1e-7,
            max_iterations=1000,
            options={"max_displacement": 0.5},
        )
        h = relentless.simulate.HOOMD(init, emin)
        h.run(pot, self.directory)

    def test_brownian_dynamics(self):
        """Test adding and removing integrator operations."""
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        brn = relentless.simulate.RunBrownianDynamics(
            steps=1, timestep=1.0e-3, T=ens.T, friction=1.0, seed=2
        )
        h = relentless.simulate.HOOMD(init, brn)
        h.run(pot, self.directory)

        brn.friction = {"A": 1.5, "B": 2.5}
        h.run(pot, self.directory)

        # temperature annealing
        brn.T = (ens.T, 1.5 * ens.T)
        h.run(pot, self.directory)

    def test_langevin_dynamics(self):
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=1, timestep=1.0e-3, T=ens.T, friction=1.0, seed=2
        )
        h = relentless.simulate.HOOMD(init, lgv)
        h.run(pot, self.directory)

        lgv.friction = {"A": 1.5, "B": 2.5}
        h.run(pot, self.directory)

        # temperature annealing
        lgv.T = (ens.T, 1.5 * ens.T)
        h.run(pot, self.directory)

    def test_molecular_dynamics(self):
        # VerletIntegrator - NVE
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        vrl = relentless.simulate.RunMolecularDynamics(steps=1, timestep=1.0e-3)
        h = relentless.simulate.HOOMD(init, vrl)
        h.run(pot, self.directory)

        # VerletIntegrator - NVT
        vrl.thermostat = relentless.simulate.NoseHooverThermostat(T=1, tau=0.5)
        h.run(pot, self.directory)

        # VerletIntegrator - NVT annealed
        vrl.thermostat.T = (1, 1.5)
        h.run(pot, self.directory)
        vrl.thermostat.T = 1

        # VerletIntegrator - NPT
        vrl.barostat = relentless.simulate.MTKBarostat(P=1, tau=0.5)
        h.run(pot, self.directory)

        # VerletIntegrator - NPH
        vrl.thermostat = None
        h.run(pot, self.directory)

        if relentless.mpi.world.size == 1:
            # VerletIntegrator - NVE (Berendsen)
            vrl.thermostat = relentless.simulate.BerendsenThermostat(T=1, tau=0.5)
            vrl.barostat = None
            h.run(pot, self.directory)

            # VerletIntegrator - incorrect
            with self.assertRaises(TypeError):
                vrl.barostat = relentless.simulate.MTKBarostat(P=1, tau=0.5)
                h.run(pot, self.directory)

    def test_temperature_ramp(self):
        if self.dim == 3:
            V = relentless.model.Cube(100.0)
        else:
            V = relentless.model.Square(100.0)
        init = relentless.simulate.InitializeRandomly(
            seed=1, N={"A": 10000}, V=V, T=2.0
        )
        logger = relentless.simulate.Record(
            filename=None,
            every=100,
            quantities=["temperature"],
        )
        brn = relentless.simulate.RunLangevinDynamics(
            steps=1000,
            timestep=1.0e-3,
            T=(2.0, 1.0),
            friction=10.0,
            seed=2,
            analyzers=logger,
        )
        h = relentless.simulate.HOOMD(init, brn)

        pot = relentless.simulate.Potentials()
        pot.pair = relentless.simulate.PairPotentialTabulator(
            None, start=0, stop=0.01, num=10, neighbor_buffer=0.5
        )
        sim = h.run(pot, self.directory)

        t = sim[logger]["timestep"]
        T = sim[logger]["temperature"]
        result = scipy.stats.linregress(x=t, y=T)
        self.assertAlmostEqual(result.intercept, 2.0, places=1)
        self.assertAlmostEqual(result.slope, -1.0 / brn.steps, places=4)

    def test_ensemble_average(self):
        """Test ensemble analyzer simulation operation."""
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, T=ens.T, diameters={"A": 1, "B": 1}
        )
        analyzer = relentless.simulate.EnsembleAverage(
            filename=None, every=5, rdf={"every": 10, "stop": 2.0, "num": 20}
        )
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=500, timestep=0.001, T=ens.T, friction=1.0, seed=1, analyzers=analyzer
        )
        h = relentless.simulate.HOOMD(init, lgv)
        sim = h.run(pot, self.directory)

        # HOOMD behavior for sampling differs between version 2 and 3
        if relentless.simulate.hoomd._hoomd_version.major >= 3:
            expected_thermo_samples = 101
            expected_rdf_samples = 51
        elif relentless.simulate.hoomd._hoomd_version.major == 2:
            expected_thermo_samples = 100
            expected_rdf_samples = 50

        # extract ensemble
        ens_ = sim[analyzer]["ensemble"]
        self.assertIsNotNone(ens_.T)
        self.assertNotEqual(ens_.T, 0)
        self.assertIsNotNone(ens_.P)
        self.assertNotEqual(ens_.P, 0)
        self.assertIsNotNone(ens_.V)
        self.assertNotEqual(ens_.V.extent, 0)
        for i, j in ens_.rdf:
            self.assertIsInstance(ens_.rdf[i, j], relentless.model.RDF)
        self.assertEqual(sim[analyzer]["num_thermo_samples"], expected_thermo_samples)
        self.assertEqual(sim[analyzer]["num_rdf_samples"], expected_rdf_samples)

        # repeat same analysis with logger, and make sure it still works
        logger = relentless.simulate.Record(
            filename=None,
            every=5,
            quantities=["temperature", "pressure"],
        )
        lgv.analyzers = logger
        sim = h.run(pot, self.directory)

        self.assertEqual(len(sim[logger]["timestep"]), expected_thermo_samples)
        self.assertEqual(len(sim[logger]["temperature"]), expected_thermo_samples)
        self.assertEqual(len(sim[logger]["pressure"]), expected_thermo_samples)
        numpy.testing.assert_array_equal(
            sim[logger]["timestep"], 5 * numpy.arange(expected_thermo_samples)
        )
        self.assertAlmostEqual(numpy.mean(sim[logger]["temperature"]), ens_.T)
        self.assertAlmostEqual(numpy.mean(sim[logger]["pressure"]), ens_.P)

    def test_ensemble_average_no_run(self):
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, T=ens.T, diameters={"A": 1, "B": 1}
        )
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=1, timestep=0.001, T=ens.T, friction=1.0, seed=1
        )
        analyzer = relentless.simulate.EnsembleAverage(
            filename=None, every=5, rdf={"every": 10, "stop": 2.0, "num": 20}
        )
        lgv2 = relentless.simulate.RunLangevinDynamics(
            steps=1, timestep=0.001, T=ens.T, friction=1.0, seed=1, analyzers=analyzer
        )
        h = relentless.simulate.HOOMD(init, [lgv, lgv2])

        # error on the ensemble average
        with self.assertRaises(RuntimeError):
            h.run(pot, self.directory)

        # error on the RDF
        lgv2.steps = 5
        with self.assertRaises(RuntimeError):
            h.run(pot, self.directory)

    def test_ensemble_average_constraints(self):
        """Test ensemble analyzer simulation operation."""
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, T=ens.T, diameters={"A": 1, "B": 1}
        )
        analyzer = relentless.simulate.EnsembleAverage(
            filename=None, every=1, assume_constraints=True
        )
        md = relentless.simulate.RunMolecularDynamics(
            steps=100,
            timestep=0.001,
            thermostat=relentless.simulate.NoseHooverThermostat(ens.T, 0.1),
            barostat=relentless.simulate.MTKBarostat(0.0, 1.0),
            analyzers=analyzer,
        )
        h = relentless.simulate.HOOMD(init, md)

        # NPT
        sim = h.run(pot, self.directory)
        ens_ = sim[analyzer]["ensemble"]
        self.assertEqual(ens_.T, ens.T)
        self.assertEqual(ens_.P, 0.0)
        self.assertNotEqual(ens_.V.extent, ens.V.extent)
        self.assertDictEqual(dict(ens_.N), dict(ens.N))
        # check that file can be saved
        if relentless.mpi.world.size == 1:
            ens_.save(self.directory.file("ensemble.json"))

        # NVT
        md.barostat = None
        sim = h.run(pot, self.directory)
        ens_ = sim[analyzer]["ensemble"]
        self.assertEqual(ens_.T, ens.T)
        self.assertNotEqual(ens_.P, 0.0)
        self.assertEqual(ens_.V.extent, ens.V.extent)
        self.assertDictEqual(dict(ens_.N), dict(ens.N))

        # NVT with annealing
        md.thermostat.T = [2 * ens.T, ens.T]
        sim = h.run(pot, self.directory)
        ens_ = sim[analyzer]["ensemble"]
        self.assertEqual(ens_.T, 1.5 * ens.T)
        self.assertNotEqual(ens_.P, 0.0)
        self.assertEqual(ens_.V.extent, ens.V.extent)
        self.assertDictEqual(dict(ens_.N), dict(ens.N))

        # NVE
        md.thermostat = None
        sim = h.run(pot, self.directory)
        ens_ = sim[analyzer]["ensemble"]
        self.assertNotEqual(ens_.T, ens.T)
        self.assertNotEqual(ens_.P, 0.0)
        self.assertEqual(ens_.V.extent, ens.V.extent)
        self.assertDictEqual(dict(ens_.N), dict(ens.N))

        # Langevin dynamics
        h.operations = relentless.simulate.RunLangevinDynamics(
            steps=100,
            timestep=0.001,
            T=ens.T,
            friction=0.1,
            seed=42,
            analyzers=analyzer,
        )
        sim = h.run(pot, self.directory)
        ens_ = sim[analyzer]["ensemble"]
        self.assertEqual(ens_.T, ens.T)
        self.assertNotEqual(ens_.P, 0.0)
        self.assertEqual(ens_.V.extent, ens.V.extent)
        self.assertDictEqual(dict(ens_.N), dict(ens.N))

        # Brownian dynamics
        h.operations = relentless.simulate.RunBrownianDynamics(
            steps=100,
            timestep=1e-5,
            T=ens.T,
            friction=0.1,
            seed=42,
            analyzers=analyzer,
        )
        sim = h.run(pot, self.directory)
        ens_ = sim[analyzer]["ensemble"]
        self.assertEqual(ens_.T, ens.T)
        self.assertNotEqual(ens_.P, 0.0)
        self.assertEqual(ens_.V.extent, ens.V.extent)
        self.assertDictEqual(dict(ens_.N), dict(ens.N))

    def test_record(self):
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, T=ens.T, diameters={"A": 1, "B": 1}
        )
        analyzer = relentless.simulate.Record(
            filename=None,
            every=5,
            quantities=[
                "potential_energy",
                "kinetic_energy",
                "temperature",
                "pressure",
            ],
        )
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=10, timestep=0.001, T=ens.T, friction=1.0, seed=1, analyzers=analyzer
        )
        h = relentless.simulate.HOOMD(init, lgv)
        sim = h.run(pot, self.directory)

        # check entries exist for data
        record_file = self.directory.file("record.dat")
        self.assertIn("timestep", sim[analyzer])
        self.assertIn("potential_energy", sim[analyzer])
        self.assertIn("kinetic_energy", sim[analyzer])
        self.assertIn("temperature", sim[analyzer])
        self.assertIn("pressure", sim[analyzer])
        self.assertFalse(os.path.exists(record_file))

        # rerun and check that arrays have same contents as file
        analyzer.filename = record_file
        sim = h.run(pot, self.directory)
        dat = relentless.mpi.world.loadtxt(record_file)
        numpy.testing.assert_allclose(dat[:, 0], sim[analyzer]["timestep"])
        numpy.testing.assert_allclose(dat[:, 1], sim[analyzer]["potential_energy"])
        numpy.testing.assert_allclose(dat[:, 2], sim[analyzer]["kinetic_energy"])
        numpy.testing.assert_allclose(dat[:, 3], sim[analyzer]["temperature"])
        numpy.testing.assert_allclose(dat[:, 4], sim[analyzer]["pressure"])

    def test_writetrajectory(self):
        """Test write trajectory simulation operation."""
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, T=ens.T, diameters={"A": 1, "B": 1}
        )
        lmpstrj = relentless.simulate.WriteTrajectory(
            filename="test_writetrajectory.lammpstrj",
            every=100,
            velocities=True,
            images=True,
            types=True,
            masses=True,
        )
        gsdtrj = relentless.simulate.WriteTrajectory(
            filename="test_writetrajectory.gsd",
            every=100,
            velocities=True,
            images=True,
            types=True,
            masses=True,
        )
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=500,
            timestep=0.001,
            T=ens.T,
            friction=1.0,
            seed=1,
            analyzers=[lmpstrj, gsdtrj],
        )
        h = relentless.simulate.HOOMD(init, lgv)
        sim = h.run(pot, self.directory)

        # read lammps trajectory file
        file = sim.directory.file(lmpstrj.filename)
        traj = lammpsio.DumpFile(file)
        for snap in traj:
            self.assertEqual(snap.N, 5)
            self.assertIsNotNone(snap.velocity)
            self.assertIsNotNone(snap.image)
            self.assertCountEqual(snap.typeid, [1, 1, 2, 2, 2])
            self.assertCountEqual(snap.mass, [1, 1, 1, 1, 1])

        # read hoomd trajectory file
        file = sim.directory.file(gsdtrj.filename)
        with gsd.hoomd.open(file) as traj:
            for snap in traj:
                self.assertEqual(snap.particles.N, 5)
                self.assertIsNotNone(snap.particles.velocity)
                self.assertCountEqual(snap.particles.typeid, [0, 0, 1, 1, 1])

    def test_self_interactions(self):
        """Test if self-interactions are excluded from rdf computation."""
        if self.dim == 3:
            Lz = 20.0
            z = 1.0
        elif self.dim == 2:
            Lz = 1.0
            z = 0.0
        else:
            raise ValueError("HOOMD supports 2d and 3d simulations")

        filename = self.directory.file("mock.gsd")
        if relentless.mpi.world.rank_is_root:
            with gsd.hoomd.open(name=filename, mode=gsd_write_mode) as f:
                s = HOOMDFrame()
                s.particles.N = 4
                s.particles.types = ["A", "B"]
                s.particles.typeid = [0, 1, 0, 1]
                s.particles.position = [
                    [-1, -1, -z],
                    [1, 1, z],
                    [1, -1, z],
                    [-1, 1, -z],
                ]
                s.configuration.box = [20, 20, Lz, 0, 0, 0]
                s.configuration.dimensions = self.dim
                f.append(s)
        relentless.mpi.world.barrier()

        _, pot = self.ens_pot()
        init = relentless.simulate.InitializeFromFile(filename=filename)
        analyzer = relentless.simulate.EnsembleAverage(
            filename=None, every=1, rdf={"stop": 2.0, "num": 20}
        )
        ig = relentless.simulate.RunMolecularDynamics(
            steps=1, timestep=0.0, analyzers=analyzer
        )
        h = relentless.simulate.HOOMD(init, ig)
        sim = h.run(pot, self.directory)

        ens_ = sim[analyzer]["ensemble"]
        for i, j in ens_.rdf:
            self.assertEqual(ens_.rdf[i, j].table[0, 1], 0.0)

    def tearDown(self):
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            self._tmp.cleanup()
            del self._tmp
        del self.directory


if __name__ == "__main__":
    unittest.main()
