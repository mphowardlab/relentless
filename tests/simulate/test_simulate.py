"""Unit tests for simulate module."""

import unittest

import numpy
from parameterized import parameterized_class

import relentless


class QuadPot(relentless.model.potential.Potential):
    """Quadratic potential function"""

    def __init__(self, types, params):
        super().__init__(types, params)

    def energy(self, key, x):
        x, u, s = self._zeros(x)
        m = self.coeff[key]["m"]
        if isinstance(m, relentless.model.IndependentVariable):
            m = m.value
        u = m * (3 - x) ** 2
        if s:
            u = u.item()
        return u

    def force(self, key, x):
        x, f, s = self._zeros(x)
        m = self.coeff[key]["m"]
        if isinstance(m, relentless.model.IndependentVariable):
            m = m.value
        f = 2 * m * (3 - x)
        if s:
            f = f.item()
        return f

    def derivative(self, key, var, x):
        x, d, s = self._zeros(x)
        if isinstance(var, relentless.model.IndependentVariable):
            if self.coeff[key]["m"] is var:
                d = (3 - x) ** 2
        if s:
            d = d.item()
        return d

    def _validate_coordinate(self, x):
        pass


class test_PotentialTabulator(unittest.TestCase):
    """Unit tests for relentless.simulate.PotentialTabulator"""

    def test_init(self):
        """Test creation with data."""
        xs = numpy.array([0.0, 0.5, 1, 1.5])
        p1 = QuadPot(types=("1",), params=("m",))

        # test creation with no potential
        t = relentless.simulate.PotentialTabulator(None, start=0.0, stop=1.5, num=4)
        numpy.testing.assert_allclose(t.linear_space, xs)
        self.assertAlmostEqual(t.start, 0.0)
        self.assertAlmostEqual(t.stop, 1.5)
        self.assertEqual(t.num, 4)
        self.assertEqual(t.potentials, [])

        # test creation with defined potential
        t = relentless.simulate.PotentialTabulator(
            potentials=p1, start=0.0, stop=1.5, num=4
        )
        numpy.testing.assert_allclose(t.linear_space, xs)
        self.assertAlmostEqual(t.start, 0.0)
        self.assertAlmostEqual(t.stop, 1.5)
        self.assertEqual(t.num, 4)
        self.assertEqual(t.potentials, [p1])

    def test_potential(self):
        """Test energy, force, and derivative methods"""
        p1 = QuadPot(types=("1",), params=("m",))
        p1.coeff["1"]["m"] = relentless.model.IndependentVariable(2.0)
        p2 = QuadPot(types=("1", "2"), params=("m",))
        for key in p2.coeff.types:
            p2.coeff[key]["m"] = 1.0
        t = relentless.simulate.PotentialTabulator(
            potentials=[p1, p2], start=0.0, stop=5.0, num=6
        )

        # test energy method
        u = t.energy("1")
        numpy.testing.assert_allclose(u, numpy.array([27, 12, 3, 0, 3, 12]))

        u = t.energy("2")
        numpy.testing.assert_allclose(u, numpy.array([9, 4, 1, 0, 1, 4]))

        # test force method
        f = t.force("1")
        numpy.testing.assert_allclose(f, numpy.array([18, 12, 6, 0, -6, -12]))

        f = t.force("2")
        numpy.testing.assert_allclose(f, numpy.array([6, 4, 2, 0, -2, -4]))

        # test derivative method
        var = p1.coeff["1"]["m"]
        d = t.derivative("1", var)
        numpy.testing.assert_allclose(d, numpy.array([9, 4, 1, 0, 1, 4]))

        d = t.derivative("2", var)
        numpy.testing.assert_allclose(d, numpy.array([0, 0, 0, 0, 0, 0]))


class QuadPairPot(relentless.model.potential.PairPotential):
    """Quadratic pair potential function"""

    def __init__(self, types):
        super().__init__(types, ("m",))

    @classmethod
    def from_json(cls, data):
        pass

    def _energy(self, r, m, **params):
        r, u, s = self._zeros(r)
        u = m * (3 - r) ** 2
        if s:
            u = u.item()
        return u

    def _force(self, r, m, **params):
        r, f, s = self._zeros(r)
        f = 2 * m * (3 - r)
        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, **params):
        r, d, s = self._zeros(r)
        if param == "m":
            d = (3 - r) ** 2
        if s:
            d = d.item()
        return d


class test_PairPotentialTabulator(unittest.TestCase):
    """Unit tests for relentless.simulate.PairPotentialTabulator"""

    def test_init(self):
        """Test creation with data."""
        rs = numpy.array([0.0, 0.5, 1, 1.5])

        # test creation with only required parameters
        t = relentless.simulate.PairPotentialTabulator(
            None, start=0.0, stop=1.5, num=4, neighbor_buffer=0.4
        )
        numpy.testing.assert_allclose(t.linear_space, rs)
        self.assertAlmostEqual(t.start, 0.0)
        self.assertAlmostEqual(t.stop, 1.5)
        self.assertEqual(t.num, 4)
        self.assertEqual(t.neighbor_buffer, 0.4)

    def test_potential(self):
        """Test energy, force, and derivative methods"""
        p1 = QuadPairPot(types=("1",))
        p1.coeff["1", "1"]["m"] = relentless.model.IndependentVariable(2.0)
        p2 = QuadPairPot(types=("1", "2"))
        for pair in p2.coeff:
            p2.coeff[pair]["m"] = 1.0
        t = relentless.simulate.PairPotentialTabulator(
            potentials=[p1, p2], start=0, stop=5, num=6, neighbor_buffer=0.4
        )

        # test energy method
        u = t.energy(("1", "1"))
        numpy.testing.assert_allclose(u, numpy.array([27, 12, 3, 0, 3, 12]) - 12)

        u = t.energy(("1", "2"))
        numpy.testing.assert_allclose(u, numpy.array([9, 4, 1, 0, 1, 4]) - 4)

        # test force method
        f = t.force(("1", "1"))
        numpy.testing.assert_allclose(f, numpy.array([18, 12, 6, 0, -6, -12]))

        f = t.force(("1", "2"))
        numpy.testing.assert_allclose(f, numpy.array([6, 4, 2, 0, -2, -4]))

        # test derivative method
        var = p1.coeff["1", "1"]["m"]
        d = t.derivative(("1", "1"), var)
        numpy.testing.assert_allclose(d, numpy.array([9, 4, 1, 0, 1, 4]) - 4)

        d = t.derivative(("1", "2"), var)
        numpy.testing.assert_allclose(d, numpy.array([0, 0, 0, 0, 0, 0]))

    def test_pairwise_compute(self):
        p1 = QuadPairPot(types=("1", "2"))
        p1.coeff["1", "1"].update(m=2.0)
        p1.coeff["1", "2"].update(m=0.0)
        p1.coeff["2", "2"].update(m=0.0)

        t = relentless.simulate.PairPotentialTabulator(
            potentials=p1, start=0.0, stop=6.0, num=7, neighbor_buffer=0.4
        )

        r, u, f = t.pairwise_energy_and_force(("1",))
        numpy.testing.assert_allclose(r, t.linear_space)
        self.assertIsInstance(u, relentless.collections.PairMatrix)
        self.assertIsInstance(f, relentless.collections.PairMatrix)

        # manually specify r
        r, u, f = t.pairwise_energy_and_force(("1",), x=t.linear_space[:-1])
        numpy.testing.assert_allclose(r, t.linear_space[:-1])

        # manually specify single r
        r, u, f = t.pairwise_energy_and_force(("1",), x=t.linear_space[0])
        self.assertEqual(r, t.linear_space[0])

        # tight without rmax
        r, u, f = t.pairwise_energy_and_force(("1",), tight=True)
        numpy.testing.assert_allclose(r, t.linear_space)

        # set rmax and use tight option
        p1.coeff["1", "1"]["rmax"] = 3.0
        r, u, f = t.pairwise_energy_and_force(("1",), tight=True)
        numpy.testing.assert_allclose(r, t.linear_space[t.linear_space <= 3.0])
        # same thing, manual r
        r, u, f = t.pairwise_energy_and_force(("1",), x=t.linear_space[:-1], tight=True)
        numpy.testing.assert_allclose(r, t.linear_space[t.linear_space <= 3.0])

        # add the second type in, but make potential all zeros
        # this will trigger the autodetect code path
        r, u, f = t.pairwise_energy_and_force(("1", "2"), tight=True)
        numpy.testing.assert_allclose(r, t.linear_space[t.linear_space <= 3.0])
        # same thing, manual r
        r, u, f = t.pairwise_energy_and_force(
            ("1", "2"), x=t.linear_space[:-1], tight=True
        )
        numpy.testing.assert_allclose(r, t.linear_space[t.linear_space <= 3.0])

        # make rmax *really* tight, but make sure we still get two points
        p1.coeff["1", "1"]["rmax"] = 1.0e-16
        r, u, f = t.pairwise_energy_and_force(("1",), tight=True, minimum_num=1)
        numpy.testing.assert_allclose(r, t.linear_space[:1])
        # same thing, manual r
        r, u, f = t.pairwise_energy_and_force(
            ("1",), x=t.linear_space[:-1], tight=True, minimum_num=1
        )
        numpy.testing.assert_allclose(r, t.linear_space[:1])
        # same thing with type 2, different minimum number
        r, u, f = t.pairwise_energy_and_force(("1", "2"), tight=True, minimum_num=2)
        numpy.testing.assert_allclose(r, t.linear_space[:2])
        # same thing, manual r
        r, u, f = t.pairwise_energy_and_force(
            ("1", "2"), x=t.linear_space[:-1], tight=True, minimum_num=2
        )
        numpy.testing.assert_allclose(r, t.linear_space[:2])

        # error for bad combination of x and tight
        with self.assertRaises(TypeError):
            t.pairwise_energy_and_force(("1",), x=1.0, tight=True)
        # error for short x with tight and minimum
        with self.assertRaises(IndexError):
            t.pairwise_energy_and_force(("1",), x=[1.0, 2.0], tight=True, minimum_num=3)


@parameterized_class(
    [
        {"box_geom": "orthorhombic", "dim": 3},
        {"box_geom": "triclinic", "dim": 3},
        {"box_geom": "orthorhombic", "dim": 2},
        {"box_geom": "triclinic", "dim": 2},
    ],
    class_name_func=lambda cls, num, params_dict: "{}_{}_{}d".format(
        cls.__name__, params_dict["box_geom"], params_dict["dim"]
    ),
)
class test_InitializeRandomly(unittest.TestCase):
    def setUp(self):
        if self.box_geom == "orthorhombic":
            if self.dim == 3:
                self.V = relentless.model.Cuboid(Lx=10, Ly=20, Lz=30)
            elif self.dim == 2:
                self.V = relentless.model.Rectangle(Lx=20, Ly=30)
        elif self.box_geom == "triclinic":
            if self.dim == 3:
                self.V = relentless.model.TriclinicBox(
                    Lx=10, Ly=20, Lz=30, xy=1, xz=2, yz=-1
                )
            elif self.dim == 2:
                self.V = relentless.model.ObliqueArea(Lx=20, Ly=30, xy=3)
        self.tol = 1.0e-8

    def test_packing_one_type(self):
        packing_fraction = {"A": 0.4}
        d = {"A": 1.2}
        N = {
            t: int(6 * phi * self.V.extent / (numpy.pi * d[t] ** 3))
            for t, phi in packing_fraction.items()
        }
        rs, types = relentless.simulate.InitializeRandomly._pack_particles(
            42, N, self.V, d
        )

        self.assertTrue(all(typei == "A" for typei in types))

        xs = self.V.coordinate_to_fraction(rs)
        self.assertTrue(numpy.all(xs >= 0))
        self.assertTrue(numpy.all(xs < 1))

        for i, r in enumerate(rs):
            dr = numpy.linalg.norm(rs[i + 1 :] - r, axis=1)
            overlaps = dr < d["A"] - self.tol
            has_overlaps = numpy.any(overlaps)
            if has_overlaps:
                print(dr[overlaps])
            self.assertFalse(has_overlaps)

    def test_packing_two_types(self):
        packing_fraction = {"A": 0.2, "B": 0.2}
        d = {"A": 1.0, "B": 3.0}
        N = {
            t: int(6 * phi * self.V.extent / (numpy.pi * d[t] ** 3))
            for t, phi in packing_fraction.items()
        }
        d["B"] = relentless.model.IndependentVariable(d["B"])
        rs, types = relentless.simulate.InitializeRandomly._pack_particles(
            42, N, self.V, d
        )

        mask = numpy.array([typei == "A" for typei in types])
        self.assertEqual(numpy.sum(mask), N["A"])
        self.assertEqual(numpy.sum(~mask), N["B"])

        xs = self.V.coordinate_to_fraction(rs)
        self.assertTrue(numpy.all(xs > -self.tol))
        self.assertTrue(numpy.all(xs < 1.0 + self.tol))

        rAs = rs[mask]
        rBs = rs[~mask]
        for i, rB in enumerate(rBs):
            dr = numpy.linalg.norm(rBs[i + 1 :] - rB, axis=1)
            overlaps = dr < d["B"].value - self.tol
            has_overlaps = numpy.any(overlaps)
            if has_overlaps:
                print(dr[overlaps])
            self.assertFalse(has_overlaps)

        for i, rA in enumerate(rAs):
            dr = numpy.linalg.norm(rAs[i + 1 :] - rA, axis=1)
            overlaps = dr < d["A"] - self.tol
            has_overlaps = numpy.any(overlaps)
            if has_overlaps:
                print(dr[overlaps])
            self.assertFalse(has_overlaps)

        for i, rA in enumerate(rAs):
            dr = numpy.linalg.norm(rBs - rA, axis=1)
            dAB = 0.5 * (d["A"] + d["B"].value)
            overlaps = dr < dAB - self.tol
            has_overlaps = numpy.any(overlaps)
            if has_overlaps:
                print(dr[overlaps])
            self.assertFalse(has_overlaps)


if __name__ == "__main__":
    unittest.main()
