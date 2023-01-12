"""Unit tests for relentless.simulate.hoomd."""
import tempfile
import unittest

import gsd.hoomd
import numpy
from parameterized import parameterized_class

import relentless
from tests.model.potential.test_pair import LinPot

_has_modules = (
    relentless.simulate.hoomd._hoomd_found and relentless.simulate.hoomd._freud_found
)


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
        self.directory = relentless.data.Directory(directory)

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
        pots.pair.potentials.append(pot)
        pots.pair.stop = 2.0
        pots.pair.num = 3

        return (ens, pots)

    # mock gsd file for testing
    def create_gsd(self):
        filename = self.directory.file("test.gsd")
        if relentless.mpi.world.rank_is_root:
            with gsd.hoomd.open(name=filename, mode="wb") as f:
                s = gsd.hoomd.Snapshot()
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

    def test_initialize_from_file(self):
        ens, pot = self.ens_pot()
        f = self.create_gsd()
        op = relentless.simulate.InitializeFromFile(filename=f)
        h = relentless.simulate.HOOMD(op)
        h.run(pot, self.directory)

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
                options={},
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
        self.assertEqual(emin.options["steps_per_iteration"], 100)

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

    def test_analyzer(self):
        """Test ensemble analyzer simulation operation."""
        ens, pot = self.ens_pot()
        init = relentless.simulate.InitializeRandomly(
            seed=1, N=ens.N, V=ens.V, T=ens.T, diameters={"A": 1, "B": 1}
        )
        analyzer = relentless.simulate.EnsembleAverage(
            check_thermo_every=5, check_rdf_every=10, rdf_dr=1.0
        )
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=500, timestep=0.001, T=ens.T, friction=1.0, seed=1, analyzers=analyzer
        )
        h = relentless.simulate.HOOMD(init, lgv)
        sim = h.run(pot, self.directory)

        # extract ensemble
        ens_ = sim[analyzer].ensemble
        self.assertIsNotNone(ens_.T)
        self.assertNotEqual(ens_.T, 0)
        self.assertIsNotNone(ens_.P)
        self.assertNotEqual(ens_.P, 0)
        self.assertIsNotNone(ens_.V)
        self.assertNotEqual(ens_.V.extent, 0)
        for i, j in ens_.rdf:
            self.assertEqual(ens_.rdf[i, j].table.shape, (len(pot.pair.x) - 1, 2))
        self.assertEqual(sim[analyzer].num_thermo_samples, 100)
        self.assertEqual(sim[analyzer].num_rdf_samples, 50)

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
            with gsd.hoomd.open(name=filename, mode="wb") as f:
                s = gsd.hoomd.Snapshot()
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
            check_thermo_every=1, check_rdf_every=1, rdf_dr=0.1
        )
        ig = relentless.simulate.RunMolecularDynamics(
            steps=1, timestep=0.0, analyzers=analyzer
        )
        h = relentless.simulate.HOOMD(init, ig)
        sim = h.run(pot, self.directory)

        ens_ = sim[analyzer].ensemble
        for i, j in ens_.rdf:
            self.assertEqual(ens_.rdf[i, j].table[0, 1], 0.0)

    def tearDown(self):
        if relentless.mpi.world.rank_is_root:
            self._tmp.cleanup()
            del self._tmp
        del self.directory


if __name__ == "__main__":
    unittest.main()
