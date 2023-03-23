"""Unit tests for objective module."""
import tempfile
import unittest

import numpy
import scipy.integrate

import relentless


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
            check_thermo_every=1, check_rdf_every=1, rdf_dr=0.1
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
        return scipy.integrate.trapz(x=r, y=y)

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

    def tearDown(self):
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            self._tmp.cleanup()
            del self._tmp
        del self.directory


if __name__ == "__main__":
    unittest.main()
