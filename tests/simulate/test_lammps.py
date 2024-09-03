"""Unit tests for relentless.simulate.lammps."""

# if being run directly, we may have arguments to handle
import argparse
import os
import pathlib
import sys

if __name__ == "__main__":
    # get optional lammps executable from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps", default=None)
    options, args = parser.parse_known_args()

    # set executable and test args
    _lammps_executable = options.lammps
    test_args = sys.argv[:1] + args

    # we also need to inject the test module onto the path when being run directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
else:
    _lammps_executable = None
    test_args = sys.argv

import tempfile
import unittest

import gsd
import gsd.hoomd
import lammpsio
import numpy
import parameterized
import scipy.stats
from packaging import version

import relentless
from tests.model.potential.test_pair import LinPot

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

# parametrize testing fixture
if _lammps_executable is not None:
    test_params = [(2, _lammps_executable), (3, _lammps_executable)]
elif relentless.simulate.lammps._lammps_found:
    test_params = [(2, None), (3, None)]
else:
    test_params = []


@unittest.skipIf(len(test_params) == 0, "No version of LAMMPS installed")
@parameterized.parameterized_class(
    ("dim", "executable"),
    test_params,
    class_name_func=lambda cls, num, params_dict: "{}_{}d-{}".format(
        cls.__name__,
        params_dict["dim"],
        "python" if params_dict["executable"] is None else "executable",
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
        self.directory = relentless.data.Directory(
            directory, create=relentless.mpi.world.rank_is_root
        )
        relentless.mpi.world.barrier()

    # mock (NVT) ensemble and potential for testing
    def ens_pot(self):
        if self.dim == 3:
            ens = relentless.model.Ensemble(
                T=2.0, V=relentless.model.Cube(L=10.0), N={"A": 2, "B": 3}
            )
        elif self.dim == 2:
            ens = relentless.model.Ensemble(
                T=2.0, V=relentless.model.Square(L=10.0), N={"A": 2, "B": 3}
            )
        else:
            raise ValueError("LAMMPS supports 2d and 3d simulations")
        ens.P = 2.5
        # setup potentials
        pot = LinPot(ens.types, params=("m",))
        for pair in pot.coeff:
            pot.coeff[pair].update({"m": -2.0, "rmax": 1.0})
        pots = relentless.simulate.Potentials()
        pots.pair = relentless.simulate.PairPotentialTabulator(
            pot, start=1e-6, stop=2.0, num=10, neighbor_buffer=0.1
        )

        return (ens, pots)

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

    def test_initialize_from_lammps_file(self):
        """Test running initialization simulation operations."""
        # InitializeFromFile
        ens, pot = self.ens_pot()
        file_ = self.create_lammps_file()
        op = relentless.simulate.InitializeFromFile(filename=file_)
        lmp = relentless.simulate.LAMMPS(
            op,
            types={"A": 1, "B": 2},
            executable=self.executable,
        )
        lmp.run(potentials=pot, directory=self.directory)

        # Run in a different directory
        with self.directory:
            d = self.directory.directory(
                "run", create=relentless.mpi.world.rank_is_root
            )
            relentless.mpi.world.barrier()
            op.filename = pathlib.Path(file_).name
            lmp.initializer = relentless.simulate.InitializeFromFile(
                pathlib.Path(file_).name
            )

            lmp.run(potentials=pot, directory=d)

        # InitializeRandomly
        op = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        lmp = relentless.simulate.LAMMPS(op, executable=self.executable)
        lmp.run(potentials=pot, directory=self.directory)

    def test_initialize_from_gsd_file(self):
        """Test running initialization simulation operations."""
        _, pot = self.ens_pot()
        file_ = self.create_gsd_file()
        op = relentless.simulate.InitializeFromFile(filename=file_)
        lmp = relentless.simulate.LAMMPS(
            op,
            types={"A": 1, "B": 2},
            executable=self.executable,
        )
        lmp.run(potentials=pot, directory=self.directory)

    def test_random_initialize_options(self):
        # no T
        ens, pot = self.ens_pot()
        op = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V)
        h = relentless.simulate.LAMMPS(op, executable=self.executable)
        h.run(potentials=pot, directory=self.directory)

        # T + diameters
        h.initializer = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, diameters={"A": 1.0, "B": 2.0}
        )
        h.run(potentials=pot, directory=self.directory)

        # no T + mass
        m = {i: idx + 1 for idx, i in enumerate(ens.N)}
        h.initializer = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, masses=m
        )
        h.run(potentials=pot, directory=self.directory)

        # T + mass
        h.initializer = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, T=ens.T, masses=m
        )
        h.run(potentials=pot, directory=self.directory)

    def test_minimization(self):
        """Test running energy minimization simulation operation."""
        ens, pot = self.ens_pot()
        file_ = self.create_lammps_file()
        init = relentless.simulate.InitializeFromFile(filename=file_)

        # MinimizeEnergy
        emin = relentless.simulate.MinimizeEnergy(
            energy_tolerance=1e-7,
            force_tolerance=1e-7,
            max_iterations=1000,
            options={"max_evaluations": 10000},
        )
        lmp = relentless.simulate.LAMMPS(
            init,
            operations=[emin],
            types={"A": 1, "B": 2},
            executable=self.executable,
        )
        lmp.run(potentials=pot, directory=self.directory)
        self.assertEqual(emin.options["max_evaluations"], 10000)

        # check default value of max_evaluations
        emin = relentless.simulate.MinimizeEnergy(
            energy_tolerance=1e-7, force_tolerance=1e-7, max_iterations=1000
        )
        lmp.operations = emin
        lmp.run(potentials=pot, directory=self.directory)

        # Whole number of iterations and evalulations as float
        emin.max_iterations = 1000.0
        emin.options = {"max_evaluations": 10000.0}
        lmp.operations = emin
        lmp.run(potentials=pot, directory=self.directory)

        # Resetting to integers
        emin.max_iterations = 1000
        emin.options = {"max_evaluations": 10000}

        # Non-whole number of iterations as float
        emin.max_iterations = 1000.5
        lmp.operations = emin
        with self.assertRaises(ValueError):
            lmp.run(potentials=pot, directory=self.directory)

        # Resetting to integer
        emin.max_iterations = 1000

        # Non-whole number of evaulations as float
        emin.options = {"max_evaluations": 10000.5}
        lmp.operations = emin
        with self.assertRaises(ValueError):
            lmp.run(potentials=pot, directory=self.directory)
        emin.options = {"max_evaluations": 10000}

    def test_brownian_dynamics(self):
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        brn = relentless.simulate.RunBrownianDynamics(
            steps=1, timestep=1.0e-3, T=ens.T, friction=1.0, seed=2
        )
        h = relentless.simulate.LAMMPS(init, brn, executable=self.executable)
        if "BROWNIAN" not in h.packages:
            self.skipTest("LAMMPS BROWNIAN package not installed")
        else:
            h.run(pot, self.directory)

        # Whole number of steps as float
        brn.steps = 1.0
        h.run(pot, self.directory)

        # Non-whole number of steps
        brn.steps = 1.5
        with self.assertRaises(ValueError):
            h.run(pot, self.directory)

        # Resetting to integer
        brn.steps = 1

        # different friction coefficients
        brn.friction = {"A": 1.5, "B": 2.5}
        h.run(pot, self.directory)

        # temperature annealing
        brn.T = (ens.T, 1.5 * ens.T)
        with self.assertRaises(NotImplementedError):
            h.run(pot, self.directory)

    def test_langevin_dynamics(self):
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        lmp = relentless.simulate.LAMMPS(init, executable=self.executable)

        # float friction
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=1, timestep=0.5, T=ens.T, friction=1.5, seed=2
        )
        lmp.operations = lgv
        lmp.run(pot, self.directory)

        # dictionary friction
        lgv.friction = {"A": 2.0, "B": 5.0}
        lmp.run(pot, self.directory)

        # single-type friction
        init_1 = relentless.simulate.InitializeRandomly(
            seed=1, N={"A": 2}, V=ens.V, T=ens.T
        )
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=1, timestep=0.5, T=ens.T, friction={"A": 3.0}, seed=2
        )
        lmp.initializer = init_1
        lmp.operations = lgv
        lmp.run(pot, self.directory)

        # temperature annealing
        lgv.T = (ens.T, 1.5 * ens.T)
        lmp.run(pot, self.directory)

        # invalid-type friction
        lgv.friction = {"B": 5.0, "C": 2.0}
        lmp.initializer = init
        lmp.operations = lgv
        with self.assertRaises(KeyError):
            lmp.run(pot, self.directory)

        # Resetting to float
        lgv.friction = 1.5

        # Whole number of steps as float
        lgv.steps = 1.0
        lmp.operations = lgv
        lmp.run(pot, self.directory)

        # Non-whole number of steps
        lgv.steps = 1.5
        lmp.operations = lgv
        with self.assertRaises(ValueError):
            lmp.run(pot, self.directory)

    def test_molecular_dynamics(self):
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(seed=1, N=ens.N, V=ens.V, T=ens.T)
        lmp = relentless.simulate.LAMMPS(init, executable=self.executable)

        # VerletIntegrator - Whole number of steps as float
        vrl = relentless.simulate.RunMolecularDynamics(steps=1.0, timestep=1e-3)
        lmp.operations = vrl
        lmp.run(pot, self.directory)

        # VerletIntegrator - Non-whole number of steps as float
        vrl.steps = 1.5
        lmp.operations = vrl
        with self.assertRaises(ValueError):
            lmp.run(pot, self.directory)

        # Resetting to integer
        vrl.steps = 1

        # VerletIntegrator - NVE
        lmp.operations = vrl
        lmp.run(pot, self.directory)

        # NVT - Berendesen
        vrl.thermostat = relentless.simulate.BerendsenThermostat(T=1, tau=0.5)
        lmp.run(pot, self.directory)

        # NVT - Berendsen annealed
        vrl.thermostat.T = (1, 1.5)
        lmp.run(pot, self.directory)
        vrl.thermostat.T = 1

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

    def test_temperature_ramp(self):
        if self.dim == 3:
            V = relentless.model.Cube(100.0)
        else:
            V = relentless.model.Square(100.0)
        init = relentless.simulate.InitializeRandomly(
            seed=1, N={"A": 20000}, V=V, T=2.0
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
        lmp = relentless.simulate.LAMMPS(init, brn, executable=self.executable)

        pot = relentless.simulate.Potentials()
        pot.pair = relentless.simulate.PairPotentialTabulator(
            None, start=1e-6, stop=0.01, num=10, neighbor_buffer=0.5
        )
        sim = lmp.run(pot, self.directory)

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
            filename=None, every=5, rdf={"stop": 2.0, "num": 20}
        )
        analyzer2 = relentless.simulate.EnsembleAverage(
            filename=None, every=10, rdf={"stop": 2.0, "num": 10}
        )
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=500,
            timestep=0.001,
            T=ens.T,
            friction=1.0,
            seed=1,
            analyzers=[analyzer, analyzer2],
        )
        h = relentless.simulate.LAMMPS(init, operations=lgv, executable=self.executable)
        sim = h.run(potentials=pot, directory=self.directory)

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

        # extract ensemble from second analyzer, answers should be slightly different
        # for any quantities that fluctuate
        ens2_ = sim[analyzer2]["ensemble"]
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
            self.assertNotEqual(ens2_.rdf[i, j].table.shape, ens_.rdf[i, j].table.shape)

        # repeat with whole number of steps as float
        analyzer.every = 5.0
        lgv.analyzers = analyzer
        sim = h.run(pot, self.directory)

        # repeat with non-whole number of steps as float
        analyzer.every = 5.5
        lgv.analyzers = analyzer
        with self.assertRaises(ValueError):
            sim = h.run(pot, self.directory)

        # repeat same analysis with logger, and make sure it still works
        logger = relentless.simulate.Record(
            filename=None,
            every=5,
            quantities=["temperature", "pressure"],
        )
        lgv.analyzers = logger
        sim = h.run(pot, self.directory)
        self.assertEqual(len(sim[logger]["timestep"]), 101)
        self.assertEqual(len(sim[logger]["temperature"]), 101)
        self.assertEqual(len(sim[logger]["pressure"]), 101)
        numpy.testing.assert_array_equal(
            sim[logger]["timestep"], numpy.linspace(0, 500, 101)
        )
        self.assertAlmostEqual(numpy.mean(sim[logger]["temperature"]), ens_.T)
        self.assertAlmostEqual(numpy.mean(sim[logger]["pressure"]), ens_.P)

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
        h = relentless.simulate.LAMMPS(init, md, executable=self.executable)

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
        if "BROWNIAN" in h.packages:
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
        h = relentless.simulate.LAMMPS(init, lgv, executable=self.executable)
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

        # repeat with whole number of steps as float
        analyzer.every = 5.0
        lgv.analyzers = analyzer
        sim = h.run(pot, self.directory)

        # repeat with non-whole number of steps as float
        analyzer.every = 5.5
        lgv.analyzers = analyzer
        with self.assertRaises(ValueError):
            sim = h.run(pot, self.directory)

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
        h = relentless.simulate.LAMMPS(init, operations=lgv, executable=self.executable)
        sim = h.run(potentials=pot, directory=self.directory)

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

        # repeat with whole number of steps as float
        lmpstrj.every = 100.0
        lgv.analyzers = [lmpstrj]
        sim = h.run(potentials=pot, directory=self.directory)

        # repeat with non-whole number of steps as float
        lmpstrj.every = 100.5
        lgv.analyzers = [lmpstrj]
        with self.assertRaises(ValueError):
            sim = h.run(potentials=pot, directory=self.directory)

    def tearDown(self):
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            self._tmp.cleanup()
            del self._tmp
        del self.directory


if __name__ == "__main__":
    unittest.main(argv=test_args)
