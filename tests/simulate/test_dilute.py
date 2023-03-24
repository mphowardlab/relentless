"""Unit tests for dilute module."""
import tempfile
import unittest

import relentless
from tests.model.potential.test_pair import LinPot


class test_Dilute(unittest.TestCase):
    """Unit tests for relentless.simulate.Dilute"""

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

    def test_run(self):
        """Test run method."""
        init = relentless.simulate.InitializeRandomly(
            seed=42, N={"A": 2, "B": 3}, V=relentless.model.Cube(L=2.0), T=1.0
        )
        analyzer = relentless.simulate.EnsembleAverage(
            every=1, rdf={"stop": 3.0, "num": 30}
        )
        md = relentless.simulate.RunMolecularDynamics(
            steps=100, timestep=1e-3, analyzers=analyzer
        )

        # set up potentials
        pot = LinPot(("A", "B"), params=("m",))
        for pair in pot.coeff:
            pot.coeff[pair]["m"] = 2.0
        pots = relentless.simulate.Potentials()
        pots.pair = relentless.simulate.PairPotentialTabulator(
            pot, start=0.0, stop=3.0, num=4, neighbor_buffer=0.5
        )

        d = relentless.simulate.Dilute(init, operations=md)
        sim = d.run(potentials=pots, directory=self.directory)
        ens_ = sim[analyzer]["ensemble"]
        self.assertAlmostEqual(ens_.P, -193.64906344)

    def test_run_moleculardynamics(self):
        """Test run molecular dynamics method with thermostat."""
        init = relentless.simulate.InitializeRandomly(
            seed=42, N={"A": 2, "B": 3}, V=relentless.model.Cube(L=2.0), T=1.0
        )
        analyzer = relentless.simulate.EnsembleAverage(
            every=1, rdf={"stop": 3.0, "num": 10}
        )
        md = relentless.simulate.RunMolecularDynamics(
            steps=100,
            timestep=1e-3,
            analyzers=analyzer,
            thermostat=relentless.simulate.BerendsenThermostat(T=2, tau=0.1),
        )

        # set up potentials
        pot = LinPot(("A", "B"), params=("m",))
        for pair in pot.coeff:
            pot.coeff[pair]["m"] = 2.0
        pots = relentless.simulate.Potentials()
        pots.pair = relentless.simulate.PairPotentialTabulator(
            pot, start=0.0, stop=3.0, num=4, neighbor_buffer=0.5
        )

        d = relentless.simulate.Dilute(init, operations=md)
        sim = d.run(potentials=pots, directory=self.directory)
        ens_ = sim[analyzer]["ensemble"]
        self.assertAlmostEqual(ens_.T, 2)

    def test_run_langevindynamics(self):
        """Test run Langevin dynamics method."""
        init = relentless.simulate.InitializeRandomly(
            seed=42, N={"A": 2, "B": 3}, V=relentless.model.Cube(L=2.0), T=1.0
        )
        analyzer = relentless.simulate.EnsembleAverage(
            every=1, rdf={"stop": 3.0, "num": 30}
        )
        lgv = relentless.simulate.RunLangevinDynamics(
            steps=100, timestep=1e-3, T=2, friction=0.1, seed=2, analyzers=analyzer
        )

        # set up potentials
        pot = LinPot(("A", "B"), params=("m",))
        for pair in pot.coeff:
            pot.coeff[pair]["m"] = 2.0
        pots = relentless.simulate.Potentials()
        pots.pair = relentless.simulate.PairPotentialTabulator(
            pot, start=0.0, stop=3.0, num=4, neighbor_buffer=0.5
        )

        d = relentless.simulate.Dilute(init, operations=lgv)
        sim = d.run(potentials=pots, directory=self.directory)
        ens_ = sim[analyzer]["ensemble"]
        self.assertAlmostEqual(ens_.T, 2)

    def test_run_browniandynamics(self):
        """Test run Brownian dynamics method."""
        init = relentless.simulate.InitializeRandomly(
            seed=42, N={"A": 2, "B": 3}, V=relentless.model.Cube(L=2.0), T=1.0
        )
        analyzer = relentless.simulate.EnsembleAverage(
            every=1, rdf={"stop": 3.0, "num": 30}
        )
        bd = relentless.simulate.RunBrownianDynamics(
            steps=100, timestep=1e-3, T=2.0, friction=0.1, seed=2, analyzers=analyzer
        )

        # set up potentials
        pot = LinPot(("A", "B"), params=("m",))
        for pair in pot.coeff:
            pot.coeff[pair]["m"] = 2.0
        pots = relentless.simulate.Potentials()
        pots.pair = relentless.simulate.PairPotentialTabulator(
            pot, start=0.0, stop=3.0, num=4, neighbor_buffer=0.5
        )

        d = relentless.simulate.Dilute(init, operations=bd)
        sim = d.run(potentials=pots, directory=self.directory)
        ens_ = sim[analyzer]["ensemble"]
        self.assertAlmostEqual(ens_.T, 2)

    def test_run_NPT(self):
        """Test run molecular dynamics method with thermostat."""
        init = relentless.simulate.InitializeRandomly(
            seed=42, N={"A": 2, "B": 3}, V=relentless.model.Cube(L=2.0), T=1.0
        )
        analyzer = relentless.simulate.EnsembleAverage(
            every=1, rdf={"stop": 3.0, "num": 30}
        )
        md = relentless.simulate.RunMolecularDynamics(
            steps=100,
            timestep=1e-3,
            analyzers=analyzer,
            thermostat=relentless.simulate.BerendsenThermostat(T=2, tau=0.1),
            barostat=relentless.simulate.Barostat(P=1),
        )

        # set up potentials
        pot = LinPot(("A", "B"), params=("m",))
        for pair in pot.coeff:
            pot.coeff[pair]["m"] = 2.0
        pots = relentless.simulate.Potentials()
        pots.pair = relentless.simulate.PairPotentialTabulator(
            pot, start=0.0, stop=3.0, num=4, neighbor_buffer=0.5
        )

        d = relentless.simulate.Dilute(init, operations=md)
        sim = d.run(potentials=pots, directory=self.directory)
        ens_ = sim[analyzer]["ensemble"]
        self.assertAlmostEqual(ens_.P, 1)
        self.assertAlmostEqual(ens_.T, 2)
        self.assertAlmostEqual(ens_.V.extent ** (1 / 3), 3.52463375)

    def test_inf_potential(self):
        """Test potential with infinite value."""
        init = relentless.simulate.InitializeRandomly(
            seed=42, N={"A": 2, "B": 3}, V=relentless.model.Cube(L=2.0), T=1.0
        )
        analyzer = relentless.simulate.EnsembleAverage(
            every=1, rdf={"stop": 3.0, "num": 30}
        )
        md = relentless.simulate.RunMolecularDynamics(
            steps=100, timestep=1e-3, analyzers=analyzer
        )

        # test with potential that has infinite potential at low r
        pot = relentless.model.potential.LennardJones(types=("A", "B"))
        for pair in pot.coeff:
            pot.coeff[pair].update(
                {"epsilon": 1.0, "sigma": 1.0, "rmax": 3.0, "shift": True}
            )
        pots = relentless.simulate.Potentials()
        pots.pair = relentless.simulate.PairPotentialTabulator(
            pot, start=0.0, stop=3.0, num=100, neighbor_buffer=0.5
        )

        d = relentless.simulate.Dilute(init, operations=md)
        warned = False
        try:
            sim = d.run(potentials=pots, directory=self.directory)
        except RuntimeWarning:
            warned = True
        self.assertFalse(warned)  # no warning should be raised
        ens_ = sim[analyzer]["ensemble"]
        self.assertAlmostEqual(ens_.P, -1.19882952)

    def tearDown(self):
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            self._tmp.cleanup()
            del self._tmp
        del self.directory


if __name__ == "__main__":
    unittest.main()
