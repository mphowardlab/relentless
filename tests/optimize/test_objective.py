"""Unit tests for objective module."""

import tempfile
import unittest

import gsd
import gsd.hoomd
import numpy
from packaging import version

import relentless

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


class QuadraticObjective(relentless.optimize.ObjectiveFunction):
    """Mock objective function used to test relentless.optimize.ObjectiveFunction"""

    def __init__(self, x):
        self.x = x

    def compute(self, variables, directory=None):
        val = (self.x.value - 1) ** 2
        variables = relentless.model.variable.graph.check_variables_and_types(
            variables, relentless.model.Variable
        )
        grad = {}
        for x in variables:
            if x is self.x:
                grad[x] = 2 * (self.x.value - 1)
            else:
                grad[x] = 0.0

        # optionally write output
        if relentless.mpi.world.rank_is_root and directory is not None:
            with open(directory.file("x.log"), "w") as f:
                f.write(str(self.x.value) + "\n")
        relentless.mpi.world.barrier()

        return relentless.optimize.ObjectiveFunctionResult(
            variables, val, grad, directory
        )


class test_ObjectiveFunction(unittest.TestCase):
    """Unit tests for relentless.optimize.ObjectiveFunction"""

    def setUp(self):
        if relentless.mpi.world.rank_is_root:
            self._tmp = tempfile.TemporaryDirectory()
            directory = self._tmp.name
        else:
            directory = None
        directory = relentless.mpi.world.bcast(directory)
        self.directory = relentless.data.Directory(directory)

    def test_compute(self):
        """Test compute method"""
        x = relentless.model.IndependentVariable(value=4.0)
        q = QuadraticObjective(x=x)

        res = q.compute(x)
        self.assertAlmostEqual(res.value, 9.0)
        self.assertAlmostEqual(res.gradient[x], 6.0)
        self.assertAlmostEqual(res.variables[x], 4.0)

        x.value = 3.0
        self.assertAlmostEqual(
            res.variables[x], 4.0
        )  # maintains the value at time of construction

        # test "invalid" variable
        with self.assertRaises(KeyError):
            res.gradient[relentless.model.variable.SameAs(x)]

    def test_directory(self):
        x = relentless.model.IndependentVariable(value=1.0)
        q = QuadraticObjective(x=x)
        d = self.directory
        q.compute(x, d)

        with open(d.file("x.log")) as f:
            x = float(f.readline())
        self.assertAlmostEqual(x, 1.0)

    def tearDown(self):
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            self._tmp.cleanup()
            del self._tmp
        del self.directory


class test_RelativeEntropy(unittest.TestCase):
    """Unit tests for relentless.optimize.RelativeEntropy"""

    def setUp(self):
        if relentless.mpi.world.rank_is_root:
            self._tmp = tempfile.TemporaryDirectory()
            directory = self._tmp.name
        else:
            directory = None
        directory = relentless.mpi.world.bcast(directory)
        self.directory = relentless.data.Directory(directory)

        lj = relentless.model.potential.LennardJones(types=("1",))
        self.epsilon = relentless.model.IndependentVariable(value=1.0)
        self.sigma = relentless.model.IndependentVariable(value=0.9)
        lj.coeff["1", "1"].update(
            {"epsilon": self.epsilon, "sigma": self.sigma, "rmax": 2.7}
        )
        self.potentials = relentless.simulate.Potentials()
        self.potentials.pair = relentless.simulate.PairPotentialTabulator(
            lj, start=0.0, stop=3.6, num=1000, neighbor_buffer=0.4
        )

        v_obj = relentless.model.Cube(L=10.0)
        self.target = relentless.model.Ensemble(T=1.5, V=v_obj, N={"1": 50})
        rs = numpy.arange(0.05, 5.0, 0.1)
        gs = numpy.exp(-lj.energy(("1", "1"), rs))
        self.target.rdf["1", "1"] = relentless.model.RDF(r=rs, g=gs)

        init = relentless.simulate.InitializeRandomly(
            seed=42, N=self.target.N, V=self.target.V, T=self.target.T
        )
        self.thermo = relentless.simulate.EnsembleAverage(
            filename="ensemble.json", every=1, rdf={"stop": 3.6, "num": 360}
        )
        md = relentless.simulate.RunMolecularDynamics(
            steps=100, timestep=1e-3, analyzers=self.thermo
        )
        self.simulation = relentless.simulate.dilute.Dilute(init, operations=md)

    def relent_grad(self, var, ext=False):
        rs = numpy.linspace(0, 3.6, 1001)[1:]
        r6_inv = numpy.power(0.9 / rs, 6)
        gs = numpy.exp(-(1 / 1.5) * 4.0 * 1.0 * (r6_inv**2 - r6_inv))
        sim_rdf = relentless.math.AkimaSpline(rs, gs)

        rs = numpy.arange(0.05, 5.0, 0.1)
        r6_inv = numpy.power(0.9 / rs, 6)
        gs = numpy.exp(-4.0 * 1.0 * (r6_inv**2 - r6_inv))
        tgt_rdf = relentless.math.AkimaSpline(rs, gs)

        rs = numpy.linspace(0, 3.6, 1001)[1:]
        r6_inv = numpy.power(0.9 / rs, 6)
        if var is self.epsilon:
            dus = 4 * (r6_inv**2 - r6_inv)
        elif var is self.sigma:
            dus = (48.0 * 1.0 / 0.9) * (r6_inv**2 - 0.5 * r6_inv)
        dudvar = relentless.math.AkimaSpline(rs, dus)

        if ext:
            norm_factor = 1.0
        else:
            norm_factor = 1000.0
        sim_factor = 50**2 * (1 / 1.5) / (1000 * norm_factor)
        tgt_factor = 50**2 * (1 / 1.5) / (1000 * norm_factor)

        r = numpy.linspace(0.05, 3.55, 1001)
        y = (
            -2
            * numpy.pi
            * r**2
            * (sim_factor * sim_rdf(r) - tgt_factor * tgt_rdf(r))
            * dudvar(r)
        )
        return relentless.math._trapezoid(x=r, y=y)

    def test_init(self):
        relent = relentless.optimize.RelativeEntropy(
            self.target, self.simulation, self.potentials, self.thermo
        )
        self.assertEqual(relent.target, self.target)
        self.assertEqual(relent.simulation, self.simulation)
        self.assertEqual(relent.potentials, self.potentials)
        self.assertEqual(relent.thermo, self.thermo)

        # test invalid target ensemble
        with self.assertRaises(ValueError):
            relent.target = relentless.model.Ensemble(T=1.5, P=1, N={"1": 50})

    def test_compute(self):
        """Test compute and compute_gradient methods"""
        relent = relentless.optimize.RelativeEntropy(
            self.target, self.simulation, self.potentials, self.thermo
        )

        res = relent.compute((self.epsilon, self.sigma))

        sim = self.simulation.run(self.potentials, self.directory)
        ensemble = sim[self.thermo]["ensemble"]
        res_grad = relent.compute_gradient(ensemble, (self.epsilon, self.sigma))

        grad_eps = self.relent_grad(self.epsilon)
        grad_sig = self.relent_grad(self.sigma)

        self.assertIsNone(res.value)
        numpy.testing.assert_allclose(res.gradient[self.epsilon], grad_eps, atol=1e-4)
        numpy.testing.assert_allclose(res.gradient[self.sigma], grad_sig, atol=1e-4)
        numpy.testing.assert_allclose(res_grad[self.epsilon], grad_eps, atol=1e-4)
        numpy.testing.assert_allclose(res_grad[self.sigma], grad_sig, atol=1e-4)

        # test extensive option
        relent = relentless.optimize.RelativeEntropy(
            self.target, self.simulation, self.potentials, self.thermo, extensive=True
        )

        res = relent.compute((self.epsilon, self.sigma))

        sim = self.simulation.run(self.potentials, self.directory)
        ensemble = sim[self.thermo]["ensemble"]
        res_grad = relent.compute_gradient(ensemble, (self.epsilon, self.sigma))

        grad_eps = self.relent_grad(self.epsilon, ext=True)
        grad_sig = self.relent_grad(self.sigma, ext=True)

        self.assertIsNone(res.value)
        numpy.testing.assert_allclose(res.gradient[self.epsilon], grad_eps, atol=1e-1)
        numpy.testing.assert_allclose(res.gradient[self.sigma], grad_sig, atol=1e-1)
        numpy.testing.assert_allclose(res_grad[self.epsilon], grad_eps, atol=1e-1)
        numpy.testing.assert_allclose(res_grad[self.sigma], grad_sig, atol=1e-1)

    def test_compute_rdf_missing(self):
        self.thermo.rdf = None
        relent = relentless.optimize.RelativeEntropy(
            self.target, self.simulation, self.potentials, self.thermo
        )
        with self.assertRaises((ValueError, KeyError)):
            relent.compute((self.epsilon, self.sigma))

    def test_compute_bonded_potential_variables(self):
        # test compute with bond potentials dependent on variables
        bond_pot = relentless.model.potential.HarmonicBond(("bondA",))
        bond_pot.coeff["bondA"].update(
            {
                "k": self.epsilon,
                "r0": 1.0,
            }
        )
        self.potentials.bond = relentless.simulate.BondPotentialTabulator(
            bond_pot, start=0.0, stop=6.0, num=1000
        )
        relent = relentless.optimize.RelativeEntropy(
            self.target, self.simulation, self.potentials, self.thermo
        )
        with self.assertRaises(ValueError):
            relent.compute((self.epsilon, self.sigma))
        self.potentials.bond = None

        # test compute with angle potentials dependent on variables
        angle_pot = relentless.model.potential.HarmonicAngle(("angleA",))
        angle_pot.coeff["angleA"].update(
            {
                "k": self.epsilon,
                "theta0": 1.5,
            }
        )
        self.potentials.angle = relentless.simulate.AnglePotentialTabulator(
            angle_pot, num=1000
        )
        relent = relentless.optimize.RelativeEntropy(
            self.target, self.simulation, self.potentials, self.thermo
        )
        with self.assertRaises(ValueError):
            relent.compute((self.epsilon, self.sigma))
        self.potentials.angle = None

        # test compute with dihedral potentials dependent on variables
        dihedral_pot = relentless.model.potential.OPLSDihedral(("dihedralA",))
        dihedral_pot.coeff["dihedralA"].update(
            {
                "k1": self.epsilon,
                "k2": 1.0,
                "k3": 1.0,
                "k4": 1.0,
            }
        )
        self.potentials.dihedral = relentless.simulate.DihedralPotentialTabulator(
            dihedral_pot, num=1000
        )
        relent = relentless.optimize.RelativeEntropy(
            self.target, self.simulation, self.potentials, self.thermo
        )
        with self.assertRaises(ValueError):
            relent.compute((self.epsilon, self.sigma))

    def test_directory(self):
        relent = relentless.optimize.RelativeEntropy(
            self.target, self.simulation, self.potentials, self.thermo
        )

        res = relent.compute((self.epsilon, self.sigma), self.directory)

        x = relentless.mpi.world.load_json(self.directory.file("pair_potential.0.json"))
        self.assertAlmostEqual(
            x["coeff"]["values"]["('1', '1')"]["epsilon"], self.epsilon.value
        )
        self.assertAlmostEqual(
            x["coeff"]["values"]["('1', '1')"]["sigma"], self.sigma.value
        )
        self.assertAlmostEqual(x["coeff"]["values"]["('1', '1')"]["rmax"], 2.7)

        z = relentless.mpi.world.load_json(self.directory.file("result.json"))
        self.assertDictEqual(
            z["variables"],
            {self.epsilon.name: self.epsilon.value, self.sigma.name: self.sigma.value},
        )
        self.assertIsNone(z["value"])
        self.assertDictEqual(
            z["gradient"],
            {
                self.epsilon.name: res.gradient[self.epsilon],
                self.sigma.name: res.gradient[self.sigma],
            },
        )
        self.assertEqual(z["directory"], self.directory.path)

        y = relentless.mpi.world.load_json(self.directory.file("ensemble.json"))
        sim = self.simulation.run(self.potentials, self.directory)
        sim_ens = sim[self.thermo]["ensemble"]
        self.assertAlmostEqual(y["T"], 1.5)
        self.assertAlmostEqual(y["N"], {"1": 50})
        self.assertAlmostEqual(y["V"]["data"], {"L": 10.0})
        self.assertAlmostEqual(y["P"], sim_ens.P)
        numpy.testing.assert_allclose(
            y["rdf"]["('1', '1')"], sim_ens.rdf["1", "1"].table.tolist()
        )


class test_RelativeEntropyDirectAverage(unittest.TestCase):
    """Unit tests for relentless.optimize.RelativeEntropy"""

    def setUp(self):
        if relentless.mpi.world.rank_is_root:
            self._tmp = tempfile.TemporaryDirectory()
            directory = self._tmp.name
        else:
            directory = None
        directory = relentless.mpi.world.bcast(directory)
        self.directory = relentless.data.Directory(directory)

        # pair potentials
        self.pair_pot = relentless.model.potential.LennardJones(("A", "B"))
        self.sigma_AA = relentless.model.IndependentVariable(value=1.1)
        self.sigma_AB = relentless.model.IndependentVariable(value=1.0)
        self.sigma_BB = relentless.model.IndependentVariable(value=0.9)
        self.pair_pot.coeff["A", "A"].update(
            {"sigma": self.sigma_AA, "epsilon": 1.0, "rmax": 6.0}
        )
        self.pair_pot.coeff["A", "B"].update(
            {"sigma": self.sigma_AB, "epsilon": 1.0, "rmax": 6.0}
        )
        self.pair_pot.coeff["B", "B"].update(
            {"sigma": self.sigma_BB, "epsilon": 1.0, "rmax": 6.0}
        )

        # bond potentials
        self.bond_pot = relentless.model.potential.HarmonicBond(("bondA", "bondB"))
        self.r0_bondA = relentless.model.IndependentVariable(value=1.0)
        self.r0_bondB = relentless.model.IndependentVariable(value=1.5)
        self.bond_pot.coeff["bondA"].update({"k": 1.0, "r0": self.r0_bondA})
        self.bond_pot.coeff["bondB"].update({"k": 1.0, "r0": self.r0_bondB})

        # angle potentials
        self.angle_pot = relentless.model.potential.HarmonicAngle(("angleA", "angleB"))
        self.theta0_angleA = relentless.model.IndependentVariable(value=1.0)
        self.theta0_angleB = relentless.model.IndependentVariable(value=1.5)
        self.angle_pot.coeff["angleA"].update({"k": 1.0, "theta0": self.theta0_angleA})
        self.angle_pot.coeff["angleB"].update({"k": 1.0, "theta0": self.theta0_angleB})

        # dihedral potentials
        self.dihedral_pot = relentless.model.potential.OPLSDihedral(("dihedralA",))
        self.k4_dihedral = relentless.model.IndependentVariable(value=0.0)
        self.dihedral_pot.coeff["dihedralA"].update(
            {"k1": 1.740, "k2": -0.157, "k3": 0.279, "k4": self.k4_dihedral}
        )

        self.potentials = relentless.simulate.Potentials()
        self.potentials.pair = relentless.simulate.PairPotentialTabulator(
            self.pair_pot, start=1e-6, stop=6.0, num=1000, neighbor_buffer=0.4
        )
        self.potentials.bond = relentless.simulate.BondPotentialTabulator(
            self.bond_pot, start=1e-6, stop=6.0, num=1000
        )
        self.potentials.angle = relentless.simulate.AnglePotentialTabulator(
            self.angle_pot, num=1000
        )
        self.potentials.dihedral = relentless.simulate.DihedralPotentialTabulator(
            self.dihedral_pot, num=1000
        )

        # target ensemble
        self.thermo = relentless.simulate.WriteTrajectory(
            filename=self.directory.file("sim.gsd"), every=1
        )

        self.simulation = None

    def create_gsd_two_4mers_sim(self, thermalize=False):
        filename = self.directory.file("sim.gsd")
        if relentless.mpi.world.rank_is_root:
            with gsd.hoomd.open(name=filename, mode="w") as f:
                s = gsd.hoomd.Frame()
                s.particles.N = 4
                s.particles.types = ["A", "B"]
                s.particles.typeid = [
                    0,
                    1,
                    0,
                    1,
                ]
                # set velocity such that T=1
                if thermalize:
                    s.particles.velocity = [
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                # bonds
                s.bonds.N = 3
                s.bonds.types = ["bondA", "bondB"]
                s.bonds.group = [(0, 1), (1, 2), (2, 3)]
                s.bonds.typeid = [0, 1, 0]
                # angles
                s.angles.N = 2
                s.angles.types = ["angleA", "angleB"]
                s.angles.group = [(0, 1, 2), (1, 2, 3)]
                s.angles.typeid = [0, 1]
                # dihedrals
                s.dihedrals.N = 1
                s.dihedrals.types = ["dihedralA"]
                s.dihedrals.group = [(0, 1, 2, 3)]
                s.dihedrals.typeid = [0]
                # box
                s.configuration.box = [20, 20, 20, 0, 0, 0]
                position_0 = [
                    [0, 0, 0],
                    [1, 1, 0],
                    [2.1, 1, 0],
                    [3.0, 0, 1],
                ]
                s.particles.position = position_0
                f.append(s)

                s1 = s
                position_1 = [
                    [0, 0, 0],
                    [numpy.sqrt(3), 1.1, 0],
                    [numpy.sqrt(3) + 1, 1.1, 0],
                    [2 * numpy.sqrt(3) + 1, 1.1, 2.2],
                ]
                s1.particles.position = position_1
                f.append(s1)
        relentless.mpi.world.barrier()
        return filename

    def create_gsd_mers_tgt(self):
        filename = self.directory.file("tgt.gsd")
        if relentless.mpi.world.rank_is_root:
            with gsd.hoomd.open(name=filename, mode="w") as f:
                s = gsd.hoomd.Frame()
                s.particles.N = 4
                s.particles.types = ["A", "B"]
                s.particles.typeid = [
                    0,
                    1,
                    0,
                    1,
                ]
                # bonds
                s.bonds.N = 3
                s.bonds.types = ["bondA", "bondB"]
                s.bonds.group = [(0, 1), (1, 2), (2, 3)]
                s.bonds.typeid = [0, 1, 0]
                # angles
                s.angles.N = 2
                s.angles.types = ["angleA", "angleB"]
                s.angles.group = [(0, 1, 2), (1, 2, 3)]
                s.angles.typeid = [0, 1]
                # dihedrals
                s.dihedrals.N = 1
                s.dihedrals.types = ["dihedralA"]
                s.dihedrals.group = [(0, 1, 2, 3)]
                s.dihedrals.typeid = [0]
                # box
                s.configuration.box = [20, 20, 20, 0, 0, 0]
                position_0 = [
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [1, 0, 0],
                ]
                s.particles.position = position_0
                f.append(s)

                s1 = s
                position_1 = [
                    [1, 0, 0],
                    [0, 1, 1],
                    [2, 1, 1],
                    [1, 2, 3 + numpy.sqrt(3)],
                ]
                s1.particles.position = position_1
                f.append(s1)
        relentless.mpi.world.barrier()
        return filename

    def test_compute_no_exclusions(self):
        """Test compute and compute_gradient methods with no exclusions"""
        self.target = self.create_gsd_mers_tgt()

        relent = relentless.optimize.RelativeEntropy(
            self.target,
            self.simulation,
            self.potentials,
            self.thermo,
            T=1.0,
            extensive=True,
        )
        sim_traj = self.create_gsd_two_4mers_sim()

        vars = (
            self.sigma_AA,
            self.sigma_BB,
            self.r0_bondA,
            self.r0_bondB,
            self.theta0_angleA,
            self.theta0_angleB,
            self.k4_dihedral,
        )
        res = relent._compute_gradient_direct_average(sim_traj, vars)

        # number of frames
        frames = 2

        # test pair contributions
        tgt_distances_AA = [1.414213562, 1.732050808]
        sim_distances_AA = [2.32594067, 2.945182781]
        s_rel_pair_sigma_AA = (
            numpy.sum(
                self.pair_pot.derivative(("A", "A"), self.sigma_AA, tgt_distances_AA)
            )
            - numpy.sum(
                self.pair_pot.derivative(("A", "A"), self.sigma_AA, sim_distances_AA)
            )
        ) / frames
        self.assertAlmostEqual(res[self.sigma_AA], s_rel_pair_sigma_AA, delta=1e-3)

        tgt_distances_BB = [1.414213562, 3.991015313]
        sim_distances_BB = [2.449489743, 3.507720287]
        s_rel_pair_sigma_BB = (
            numpy.sum(
                self.pair_pot.derivative(("B", "B"), self.sigma_BB, tgt_distances_BB)
            )
            - numpy.sum(
                self.pair_pot.derivative(("B", "B"), self.sigma_BB, sim_distances_BB)
            )
        ) / frames
        self.assertAlmostEqual(res[self.sigma_BB], s_rel_pair_sigma_BB, delta=1e-3)

        # test bond contributions
        tgt_distances_bondA = [1, 1, 1.732050808, 3.991015313]
        sim_distances_bondA = [1.414213562, 1.676305461, 2.051828453, 2.800060861]
        s_rel_bond_r0_bondA = (
            numpy.sum(
                self.bond_pot.derivative("bondA", self.r0_bondA, tgt_distances_bondA)
            )
            - numpy.sum(
                self.bond_pot.derivative("bondA", self.r0_bondA, sim_distances_bondA)
            )
        ) / frames
        self.assertAlmostEqual(res[self.r0_bondA], s_rel_bond_r0_bondA, delta=1e-3)

        tgt_distances_bondB = [1.0, 2.0]
        sim_distances_bondB = [1.1, 1.0]
        s_rel_bond_r0_bondB = (
            numpy.sum(
                self.bond_pot.derivative("bondB", self.r0_bondB, tgt_distances_bondB)
            )
            - numpy.sum(
                self.bond_pot.derivative("bondB", self.r0_bondB, sim_distances_bondB)
            )
        ) / frames
        self.assertAlmostEqual(res[self.r0_bondB], s_rel_bond_r0_bondB, delta=1e-3)

        # test angle contributions
        tgt_angle_angleA = [1.570796327, 0.955316618]
        sim_angle_angleA = [2.35619449, 2.57577384]
        s_rel_angle_theta0_angleA = (
            numpy.sum(
                self.angle_pot.derivative(
                    "angleA", self.theta0_angleA, tgt_angle_angleA
                )
            )
            - numpy.sum(
                self.angle_pot.derivative(
                    "angleA", self.theta0_angleA, sim_angle_angleA
                )
            )
        ) / frames
        self.assertAlmostEqual(
            res[self.theta0_angleA], s_rel_angle_theta0_angleA, delta=1e-3
        )

        tgt_angle_angleB = [1.570796327, 1.317534763]
        sim_angle_angleB = [2.137548653, 2.237742671]
        s_rel_angle_theta0_angleB = (
            numpy.sum(
                self.angle_pot.derivative(
                    "angleB", self.theta0_angleB, tgt_angle_angleB
                )
            )
            - numpy.sum(
                self.angle_pot.derivative(
                    "angleB", self.theta0_angleB, sim_angle_angleB
                )
            )
        ) / frames
        self.assertAlmostEqual(
            res[self.theta0_angleB], s_rel_angle_theta0_angleB, delta=1e-3
        )

        # test dihedral contributions
        tgt_dihedral = [0, 0.523598776]
        sim_dihedral = [0.785398, 1.570796327]
        s_rel_dihedral_phi0 = (
            numpy.sum(
                self.dihedral_pot.derivative(
                    "dihedralA", self.k4_dihedral, tgt_dihedral
                )
            )
            - numpy.sum(
                self.dihedral_pot.derivative(
                    "dihedralA", self.k4_dihedral, sim_dihedral
                )
            )
        ) / frames
        self.assertAlmostEqual(res[self.k4_dihedral], s_rel_dihedral_phi0, delta=1e-3)

    def test_compute_1_2_exclusions(self):
        """Test compute and compute_gradient methods with 1-2 exclusions"""
        self.target = self.create_gsd_mers_tgt()

        # add 1-2 exclusions to the pair potential
        self.potentials.pair.exclusions = ["1-2"]

        relent = relentless.optimize.RelativeEntropy(
            self.target,
            self.simulation,
            self.potentials,
            self.thermo,
            T=1.0,
            extensive=True,
        )
        sim_traj = self.create_gsd_two_4mers_sim()

        vars = (
            self.sigma_AA,
            self.sigma_AB,
            self.sigma_BB,
        )
        res = relent._compute_gradient_direct_average(sim_traj, vars)

        # number of frames
        frames = 2

        # test A-A pair contributions
        tgt_distances_AA = [1.414213562, 1.732050808]
        sim_distances_AA = [2.32594067, 2.945182781]
        s_rel_pair_sigma_AA = (
            numpy.sum(
                self.pair_pot.derivative(("A", "A"), self.sigma_AA, tgt_distances_AA)
            )
            - numpy.sum(
                self.pair_pot.derivative(("A", "A"), self.sigma_AA, sim_distances_AA)
            )
        ) / frames
        self.assertAlmostEqual(res[self.sigma_AA], s_rel_pair_sigma_AA, delta=1e-3)

        # test A-B pair contributions
        tgt_distances_AB = [1.0, 5.137344143]
        sim_distances_AB = [3.16227766, 5.096881716]
        s_rel_pair_sigma_AB = (
            numpy.sum(
                self.pair_pot.derivative(("A", "B"), self.sigma_AB, tgt_distances_AB)
            )
            - numpy.sum(
                self.pair_pot.derivative(("A", "B"), self.sigma_AB, sim_distances_AB)
            )
        ) / frames
        self.assertAlmostEqual(res[self.sigma_AB], s_rel_pair_sigma_AB, delta=1e-3)

        # test B-B pair contributions
        tgt_distances_BB = [1.414213562, 3.991015313]
        sim_distances_BB = [2.449489743, 3.507720287]
        s_rel_pair_sigma_BB = (
            numpy.sum(
                self.pair_pot.derivative(("B", "B"), self.sigma_BB, tgt_distances_BB)
            )
            - numpy.sum(
                self.pair_pot.derivative(("B", "B"), self.sigma_BB, sim_distances_BB)
            )
        ) / frames
        self.assertAlmostEqual(res[self.sigma_BB], s_rel_pair_sigma_BB, delta=1e-3)

    def test_compute_1_3_exclusions(self):
        """Test compute and compute_gradient methods with 1-3 exclusions"""
        self.target = self.create_gsd_mers_tgt()

        # add 1-3 exclusions to the pair potential
        self.potentials.pair.exclusions = ["1-3"]

        relent = relentless.optimize.RelativeEntropy(
            self.target,
            self.simulation,
            self.potentials,
            self.thermo,
            T=1.0,
            extensive=True,
        )
        sim_traj = self.create_gsd_two_4mers_sim()

        vars = (
            self.sigma_AA,
            self.sigma_AB,
            self.sigma_BB,
        )
        res = relent._compute_gradient_direct_average(sim_traj, vars)

        # number of frames
        frames = 2

        # test A-A pair contributions
        self.assertAlmostEqual(res[self.sigma_AA], 0.0, delta=1e-3)

        # test A-B pair contributions
        tgt_distances_AB = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.732050808,
            5.137344143,
            2.0,
            3.991015313,
        ]
        sim_distances_AB = [
            1.414213562,
            3.16227766,
            1.1,
            1.676305461,
            2.051828453,
            5.096881716,
            1.0,
            2.8,
        ]
        s_rel_pair_sigma_AB = (
            numpy.sum(
                self.pair_pot.derivative(("A", "B"), self.sigma_AB, tgt_distances_AB)
            )
            - numpy.sum(
                self.pair_pot.derivative(("A", "B"), self.sigma_AB, sim_distances_AB)
            )
        ) / frames
        self.assertAlmostEqual(res[self.sigma_AB], s_rel_pair_sigma_AB, delta=1e-3)

        # test B-B pair contributions
        self.assertAlmostEqual(res[self.sigma_BB], 0.0, delta=1e-3)

    def test_compute_1_4_exclusions(self):
        """Test compute and compute_gradient methods with 1-4 exclusions"""
        self.target = self.create_gsd_mers_tgt()

        # add 1-4 exclusions to the pair potential
        self.potentials.pair.exclusions = ["1-4"]

        relent = relentless.optimize.RelativeEntropy(
            self.target,
            self.simulation,
            self.potentials,
            self.thermo,
            T=1.0,
            extensive=True,
        )
        sim_traj = self.create_gsd_two_4mers_sim()

        vars = (
            self.sigma_AA,
            self.sigma_AB,
            self.sigma_BB,
        )
        res = relent._compute_gradient_direct_average(sim_traj, vars)

        # number of frames
        frames = 2

        # test A-A pair contributions
        tgt_distances_AA = [1.414213562, 1.732050808]
        sim_distances_AA = [2.32594067, 2.945182781]
        s_rel_pair_sigma_AA = (
            numpy.sum(
                self.pair_pot.derivative(("A", "A"), self.sigma_AA, tgt_distances_AA)
            )
            - numpy.sum(
                self.pair_pot.derivative(("A", "A"), self.sigma_AA, sim_distances_AA)
            )
        ) / frames
        self.assertAlmostEqual(res[self.sigma_AA], s_rel_pair_sigma_AA, delta=1e-3)

        # test A-B pair contributions
        tgt_distances_AB = [1.0, 1.0, 1.0, 1.732050808, 2.0, 3.991015313]
        sim_distances_AB = [1.414213562, 1.1, 1.676305461, 2.051828453, 1.0, 2.8]
        s_rel_pair_sigma_AB = (
            numpy.sum(
                self.pair_pot.derivative(("A", "B"), self.sigma_AB, tgt_distances_AB)
            )
            - numpy.sum(
                self.pair_pot.derivative(("A", "B"), self.sigma_AB, sim_distances_AB)
            )
        ) / frames
        self.assertAlmostEqual(res[self.sigma_AB], s_rel_pair_sigma_AB, delta=1e-3)

        # test B-B pair contributions
        tgt_distances_BB = [1.414213562, 3.991015313]
        sim_distances_BB = [2.449489743, 3.507720287]
        s_rel_pair_sigma_BB = (
            numpy.sum(
                self.pair_pot.derivative(("B", "B"), self.sigma_BB, tgt_distances_BB)
            )
            - numpy.sum(
                self.pair_pot.derivative(("B", "B"), self.sigma_BB, sim_distances_BB)
            )
        ) / frames
        self.assertAlmostEqual(res[self.sigma_BB], s_rel_pair_sigma_BB, delta=1e-3)

    def test_compute_all_exclusions(self):
        """Test compute and compute_gradient methods with all exclusions"""
        self.target = self.create_gsd_mers_tgt()

        # add 1-2, 1-3, and 1-4 exclusions to the pair potential
        self.potentials.pair.exclusions = ["1-2", "1-3", "1-4"]

        relent = relentless.optimize.RelativeEntropy(
            self.target,
            self.simulation,
            self.potentials,
            self.thermo,
            T=1.0,
            extensive=True,
        )
        sim_traj = self.create_gsd_two_4mers_sim()

        vars = (
            self.sigma_AA,
            self.sigma_AB,
            self.sigma_BB,
        )
        res = relent._compute_gradient_direct_average(sim_traj, vars)

        # test A-A pair contributions
        self.assertAlmostEqual(res[self.sigma_AA], 0.0, delta=1e-3)

        # test A-B pair contributions
        self.assertAlmostEqual(res[self.sigma_AB], 0.0, delta=1e-3)

        # test B-B pair contributions
        self.assertAlmostEqual(res[self.sigma_BB], 0.0, delta=1e-3)

    def test_intensive(self):
        """Test compute and compute_gradient methods"""
        self.target = self.create_gsd_mers_tgt()

        relent = relentless.optimize.RelativeEntropy(
            self.target,
            self.simulation,
            self.potentials,
            self.thermo,
            T=1.0,
            extensive=True,
        )
        sim_traj = self.create_gsd_two_4mers_sim()

        vars = (
            self.sigma_AA,
            self.sigma_BB,
            self.r0_bondA,
            self.r0_bondB,
            self.theta0_angleA,
            self.theta0_angleB,
            self.k4_dihedral,
        )
        res_extensive = relent._compute_gradient_direct_average(sim_traj, vars)

        relentl_intensive = relentless.optimize.RelativeEntropy(
            self.target,
            self.simulation,
            self.potentials,
            self.thermo,
            T=1.0,
            extensive=False,
        )
        res_intensive = relentl_intensive._compute_gradient_direct_average(
            sim_traj, vars
        )

        for var in vars:
            self.assertAlmostEqual(
                res_extensive[var], res_intensive[var] * 20**3, delta=1e-3
            )

    def test_compute_temperature(self):
        """Test compute and compute_gradient methods with 1-2 exclusions"""
        # check case where temperature cannot be computed
        self.target = self.create_gsd_mers_tgt()

        self.potentials.pair.exclusions = ["1-2"]

        relent = relentless.optimize.RelativeEntropy(
            self.target,
            self.simulation,
            self.potentials,
            self.thermo,
            extensive=True,
        )
        sim_traj = self.create_gsd_two_4mers_sim()

        vars = (
            self.sigma_AA,
            self.sigma_AB,
            self.sigma_BB,
        )

        with self.assertRaises(ValueError):
            res = relent._compute_gradient_direct_average(sim_traj, vars)

        # check case where temperature can be computed
        self.sim = self.create_gsd_two_4mers_sim(thermalize=True)
        relent = relentless.optimize.RelativeEntropy(
            self.target,
            self.simulation,
            self.potentials,
            self.thermo,
            extensive=True,
        )

        res = relent._compute_gradient_direct_average(sim_traj, vars)

        # number of frames
        frames = 2

        # test A-A pair contributions
        tgt_distances_AA = [1.414213562, 1.732050808]
        sim_distances_AA = [2.32594067, 2.945182781]
        s_rel_pair_sigma_AA = (
            numpy.sum(
                self.pair_pot.derivative(("A", "A"), self.sigma_AA, tgt_distances_AA)
            )
            - numpy.sum(
                self.pair_pot.derivative(("A", "A"), self.sigma_AA, sim_distances_AA)
            )
        ) / frames
        self.assertAlmostEqual(res[self.sigma_AA], s_rel_pair_sigma_AA, delta=1e-3)

    def tearDown(self):
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            self._tmp.cleanup()
            del self._tmp
        del self.directory


if __name__ == "__main__":
    unittest.main()
