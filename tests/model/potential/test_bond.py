"""Unit tests for bond module."""

import tempfile
import unittest

import numpy

import relentless


class LinPotBond(relentless.model.potential.bond.BondPotential):
    """Linear bond potential function"""

    def __init__(self, types, params):
        super().__init__(types, params)

    def to_json(self):
        data = super().to_json()
        data["params"] = self.coeff.params
        return data

    def energy(self, types, r):
        m = self.coeff[types]["m"]

        r, u, s = self._zeros(r)
        u[:] = m * r
        if s:
            u = u.item()
        return u

    def force(self, types, r):
        m = self.coeff[types]["m"]

        r, f, s = self._zeros(r)
        f[:] = -m
        if s:
            f = f.item()
        return f

    def _derivative(self, param, r, **params):
        r, d, s = self._zeros(r)
        if param == "m":
            d[:] = r
        if s:
            d = d.item()
        return d


class TwoVarPot(relentless.model.potential.bond.BondPotential):
    """Mock potential function"""

    def __init__(self, types, params):
        super().__init__(types, params)

    @classmethod
    def from_json(cls, data):
        raise NotImplementedError()

    def energy(self, r, x, y, **params):
        pass

    def force(self, r, x, y, **params):
        pass

    def _derivative(self, param, r, **params):
        # not real derivative, just used to test functionality
        r, d, s = self._zeros(r)
        if param == "x":
            d[:] = 2 * r
        elif param == "y":
            d[:] = 3 * r
        if s:
            d = d.item()
        return d


class test_BondPotential(unittest.TestCase):
    """Unit tests for relentless.model.bond.BondPotential"""

    def test_init(self):
        """Test creation from data"""
        # test creation with only m
        p = LinPotBond(types=("1",), params=("m",))
        p.coeff["1"]["m"] = 3.5

        coeff = relentless.model.potential.bond.BondParameters(
            types=("1",), params=("m")
        )
        coeff["1"]["m"] = 3.5

        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertEqual(p.coeff.evaluate(("1")), coeff.evaluate(("1")))

    def test_energy(self):
        """Test energy method"""
        p = LinPotBond(types=("1",), params=("m",))
        p.coeff["1"]["m"] = 2.0

        # test with no cutoffs
        u = p.energy(types=("1"), r=0.5)
        self.assertAlmostEqual(u, 1.0)
        u = p.energy(types=("1"), r=[0.25, 0.75])
        numpy.testing.assert_allclose(u, [0.5, 1.5])

    def test_force(self):
        """Test force method"""
        p = LinPotBond(types=("1",), params=("m",))
        p.coeff["1"]["m"] = 2.0

        # test with no cutoffs
        f = p.force(types=("1"), r=0.5)
        self.assertAlmostEqual(f, -2.0)
        f = p.force(types=("1"), r=[0.25, 0.75])
        numpy.testing.assert_allclose(f, [-2.0, -2.0])

    def test_derivative_values(self):
        """Test derivative method with different param values"""
        p = LinPotBond(types=("1",), params=("m",))
        x = relentless.model.IndependentVariable(value=2.0)
        p.coeff["1"]["m"] = x

        # test with no cutoffs
        d = p.derivative(type_=("1"), var=x, r=0.5)
        self.assertAlmostEqual(d, 0.5)
        d = p.derivative(type_=("1"), var=x, r=[0.25, 0.75])
        numpy.testing.assert_allclose(d, [0.25, 0.75])

    def test_derivative_types(self):
        """Test derivative method with different param types."""
        q = LinPotBond(types=("1",), params=("m",))
        x = relentless.model.IndependentVariable(value=4.0)
        y = relentless.model.IndependentVariable(value=64.0)
        z = relentless.model.GeometricMean(x, y)
        q.coeff["1"]["m"] = z

        # test with respect to dependent variable parameter
        d = q.derivative(type_=("1"), var=z, r=2.0)
        self.assertAlmostEqual(d, 2.0)

        # test with respect to independent variable on which parameter is dependent
        d = q.derivative(type_=("1"), var=x, r=1.5)
        self.assertAlmostEqual(d, 3.0)
        d = q.derivative(type_=("1"), var=y, r=4.0)
        self.assertAlmostEqual(d, 0.5)

        # test invalid derivative w.r.t. scalar
        a = 2.5
        q.coeff["1"]["m"] = a
        with self.assertRaises(TypeError):
            d = q.derivative(type_=("1"), var=a, r=2.0)

        # test with respect to independent variable which is
        # related to a SameAs variable
        r = TwoVarPot(types=("1",), params=("x", "y"))

        r.coeff["1"]["x"] = x
        r.coeff["1"]["y"] = relentless.model.variable.SameAs(x)
        d = r.derivative(type_=("1"), var=x, r=4.0)
        self.assertAlmostEqual(d, 20.0)

        r.coeff["1"]["y"] = x
        r.coeff["1"]["x"] = relentless.model.variable.SameAs(x)
        d = r.derivative(type_=("1"), var=x, r=4.0)
        self.assertAlmostEqual(d, 20.0)

    def test_iteration(self):
        """Test iteration on typesPotential object"""
        p = LinPotBond(types=("1",), params=("m",))
        for types in p.coeff:
            p.coeff[types]["m"] = 2.0

        self.assertEqual(dict(p.coeff["1"]), {"m": 2.0})

    def test_json(self):
        """Test saving to file"""
        p = LinPotBond(types=("1",), params=("m"))
        p.coeff["1"]["m"] = 2.0

        data = p.to_json()
        self.assertEqual(data["id"], p.id)
        self.assertEqual(data["name"], p.name)

        p2 = LinPotBond.from_json(data)
        self.assertEqual(p2.coeff["1"]["m"], 2.0)

    def test_save(self):
        """Test saving to file"""
        temp = tempfile.NamedTemporaryFile()
        p = LinPotBond(types=("1",), params=("m"))
        p.coeff["1"]["m"] = 2.0
        p.save(temp.name)

        p2 = LinPotBond.from_file(temp.name)
        self.assertEqual(p2.coeff["1"]["m"], 2.0)

        temp.close()


class test_HarmonicBond(unittest.TestCase):
    """Unit tests for relentless.model.potential.HarmonicBond"""

    def test_init(self):
        """Test creation from data"""
        harmonic_bond = relentless.model.potential.HarmonicBond(types=("1",))
        coeff = relentless.model.potential.BondParameters(
            types=("1",),
            params=(
                "k",
                "r0",
            ),
        )
        self.assertCountEqual(harmonic_bond.coeff.types, coeff.types)
        self.assertCountEqual(harmonic_bond.coeff.params, coeff.params)

    def test_energy(self):
        """Test _energy method"""
        harmonic_bond = relentless.model.potential.HarmonicBond(types=("1",))
        harmonic_bond.coeff["1"].update(k=1000, r0=1.0)
        # test scalar r
        r_input = 0.5
        u_actual = 125.0
        u = harmonic_bond.energy(type_=("1"), r=r_input)
        self.assertAlmostEqual(u, u_actual)

        # test array r
        r_input = numpy.array([0.0, 0.5, 1.0])
        u_actual = numpy.array([500.0, 125.0, 0.0])
        u = harmonic_bond.energy(type_=("1"), r=r_input)
        numpy.testing.assert_allclose(u, u_actual)

    def test_force(self):
        """Test _force method"""
        harmonic_bond = relentless.model.potential.HarmonicBond(types=("1",))
        harmonic_bond.coeff["1"].update(k=1000, r0=1.0)

        # test scalar r
        r_input = 0.5
        f_actual = 500
        f = harmonic_bond.force(type_=("1"), r=r_input)
        self.assertAlmostEqual(f, f_actual)

        # test array r
        r_input = numpy.array([0.0, 0.5, 1.0])
        f_actual = numpy.array([1000, 500, 0])
        f = harmonic_bond.force(type_=("1"), r=r_input)
        numpy.testing.assert_allclose(f, f_actual)

    def test_derivative(self):
        """Test _derivative method"""
        harmonic_bond = relentless.model.potential.HarmonicBond(types=("1",))

        # w.r.t. k
        # test scalar r
        r_input = 0.5
        d_actual = 0.125
        d = harmonic_bond._derivative(param="k", r=r_input, k=1000, r0=1.0)
        self.assertAlmostEqual(d, d_actual)

        # test array r
        r_input = numpy.array([0, 1, 1.5])
        d_actual = numpy.array([0.50, 0.0, 0.125])
        d = harmonic_bond._derivative(param="k", r=r_input, k=1000, r0=1.0)
        numpy.testing.assert_allclose(d, d_actual)

        # w.r.t. r0
        # test scalar r
        r_input = 0.5
        d_actual = 500
        d = harmonic_bond._derivative(param="r0", r=r_input, k=1000.0, r0=1.0)
        self.assertAlmostEqual(d, d_actual)

        # test array r
        r_input = numpy.array([0, 1, 1.5])
        d_actual = numpy.array([1000, 0, -500])
        d = harmonic_bond._derivative(param="r0", r=r_input, k=1000.0, r0=1.0)
        numpy.testing.assert_allclose(d, d_actual)

        # test invalid param
        with self.assertRaises(ValueError):
            harmonic_bond._derivative(param="ro", r=r_input, k=1000.0, r0=1.0)


class test_FENEWCA(unittest.TestCase):
    """Unit tests for relentless.model.potential.FENEWCA"""

    def test_init(self):
        """Test creation from data"""
        FENEWCA = relentless.model.potential.FENEWCA(types=("1",))
        coeff = relentless.model.potential.BondParameters(
            types=("1",),
            params=(
                "k",
                "r0",
                "epsilon",
                "sigma",
            ),
        )
        self.assertCountEqual(FENEWCA.coeff.types, coeff.types)
        self.assertCountEqual(FENEWCA.coeff.params, coeff.params)

    def test_energy(self):
        """Test _energy method"""
        FENEWCA = relentless.model.potential.FENEWCA(types=("1",))
        FENEWCA.coeff["1"].update(k=30, r0=1.5, epsilon=1.0, sigma=1.0)

        # test scalar r
        r_input = 0.95
        u_actual = 20.2638974009
        u = FENEWCA.energy(type_=("1"), r=r_input)
        self.assertAlmostEqual(u, u_actual)

        # test array r
        r_input = numpy.array([0, 0.9, 1.2, 2])
        u_actual = numpy.array([numpy.inf, 22.698308667, 34.4807296042, numpy.inf])
        u = FENEWCA.energy(type_=("1"), r=r_input)
        numpy.testing.assert_allclose(u, u_actual)

    def test_force(self):
        """Test _force method"""
        FENEWCA = relentless.model.potential.FENEWCA(types=("1",))
        FENEWCA.coeff["1"].update(k=30, r0=1.5, epsilon=1.0, sigma=1.0)

        # test scalar r
        r_input = 0.95
        f_actual = 11.549426778
        f = FENEWCA.force(type_=("1"), r=r_input)
        numpy.testing.assert_allclose(f, f_actual)

        # test array r
        r_input = numpy.array([0, 0.9, 1.2, 2])
        f_actual = numpy.array([numpy.inf, 96.4721239943, -100, numpy.inf])
        f = FENEWCA.force(type_=("1"), r=r_input)
        numpy.testing.assert_allclose(f, f_actual)

    def test_derivative(self):
        """Test _derivative method"""
        FENEWCA = relentless.model.potential.FENEWCA(types=("1",))

        # w.r.t. k
        # test scalar r
        r_input = 0.95
        d_actual = -0.576764091467
        d = FENEWCA._derivative(param="k", r=r_input, k=30, r0=1.5, epsilon=1, sigma=1)
        self.assertAlmostEqual(d, d_actual)

        # test array r
        r_input = numpy.array([0, 1, 1.6])
        d_actual = numpy.array([0, -0.661259998015, numpy.inf])
        d = FENEWCA._derivative(param="k", r=r_input, k=30, r0=1.5, epsilon=1, sigma=1)
        numpy.testing.assert_allclose(d, d_actual)

        # w.r.t. r0
        # test scalar r
        r_input = 0.95
        d_actual = 7.06858290903
        d = FENEWCA._derivative(param="r0", r=r_input, k=30, r0=1.5, epsilon=1, sigma=1)
        self.assertAlmostEqual(d, d_actual)

        # test array r
        r_input = numpy.array([0, 1, 1.6])
        d_actual = numpy.array([0, 9.5496000794, numpy.inf])
        d = FENEWCA._derivative(param="r0", r=r_input, k=30, r0=1.5, epsilon=1, sigma=1)
        numpy.testing.assert_allclose(d, d_actual)

        # w.r.t. epsilon
        # test scalar r
        r_input = 0.95
        d_actual = 2.96097465689
        d = FENEWCA._derivative(
            param="epsilon", r=r_input, k=30, r0=1.5, epsilon=1, sigma=1
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        r_input = numpy.array([0, 1.0, 1.1, 2.0])
        d_actual = numpy.array([numpy.inf, 1.0, 0.0166275506263, 0])
        d = FENEWCA._derivative(
            param="epsilon", r=r_input, k=30, r0=1.5, epsilon=1, sigma=1
        )

        # test array r
        r_input = numpy.array([0, 1, 1.6])
        d_actual = numpy.array([0, 9.5496000794, numpy.inf])
        d = FENEWCA._derivative(param="r0", r=r_input, k=30, r0=1.5, epsilon=1, sigma=1)
        numpy.testing.assert_allclose(d, d_actual)

        # w.r.t. sigma
        # test scalar r
        r_input = 0.95
        d_actual = 56.1806752906
        d = FENEWCA._derivative(
            param="sigma", r=r_input, k=30, r0=1.5, epsilon=1, sigma=1
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        r_input = numpy.array([0, 1.0, 1.1, 2.0])
        d_actual = numpy.array([numpy.inf, 24, 1.74690492881, 0])
        d = FENEWCA._derivative(
            param="sigma", r=r_input, k=30, r0=1.5, epsilon=1, sigma=1
        )
        numpy.testing.assert_allclose(d, d_actual)

        # test invalid param
        with self.assertRaises(ValueError):
            FENEWCA._derivative(
                param="sgima", r=r_input, k=30, r0=1.5, epsilon=1, sigma=1
            )

    def test_json(self):
        FENEWCA = relentless.model.potential.FENEWCA(types=("1",))
        FENEWCA.coeff["1"].update(k=1000, r0=1.0, epsilon=1.0, sigma=1.0)
        data = FENEWCA.to_json()

        FENEWCA2 = relentless.model.potential.FENEWCA.from_json(data)
        self.assertEqual(FENEWCA2.coeff["1"]["k"], 1000)
        self.assertEqual(FENEWCA2.coeff["1"]["r0"], 1.0)


class test_BondSpline(unittest.TestCase):
    """Unit tests for relentless.model.potential.BondSpline"""

    def test_init(self):
        """Test creation from data"""
        # test diff mode
        s = relentless.model.potential.BondSpline(types=("1",), num_knots=3)
        self.assertEqual(s.num_knots, 3)
        self.assertEqual(s.mode, "diff")
        coeff = relentless.model.potential.BondParameters(
            types=("1",),
            params=(
                "dr-0",
                "dr-1",
                "dr-2",
                "diff-0",
                "diff-1",
                "diff-2",
            ),
        )
        self.assertCountEqual(s.coeff.types, coeff.types)
        self.assertCountEqual(s.coeff.params, coeff.params)

        # test value mode
        s = relentless.model.potential.BondSpline(
            types=("1",), num_knots=3, mode="value"
        )
        self.assertEqual(s.num_knots, 3)
        self.assertEqual(s.mode, "value")
        coeff = relentless.model.potential.BondParameters(
            types=("1",),
            params=(
                "dr-0",
                "dr-1",
                "dr-2",
                "value-0",
                "value-1",
                "value-2",
            ),
        )
        self.assertCountEqual(s.coeff.types, coeff.types)
        self.assertCountEqual(s.coeff.params, coeff.params)

        # test invalid number of knots
        with self.assertRaises(ValueError):
            s = relentless.model.potential.BondSpline(types=("1",), num_knots=1)

        # test invalid mode
        with self.assertRaises(ValueError):
            s = relentless.model.potential.BondSpline(
                types=("1",), num_knots=3, mode="val"
            )

    def test_from_array(self):
        """Test from_array method and knots generator"""
        r_arr = [1, 2, 3]
        u_arr = [9, 4, 1]
        u_arr_diff = [5, 3, 1]

        # test diff mode
        s = relentless.model.potential.BondSpline(types=("1",), num_knots=3)
        s.from_array(types=("1"), r=r_arr, u=u_arr)

        dvars = []
        for i, (r, k) in enumerate(s.knots(types=("1"))):
            self.assertAlmostEqual(r.value, 1.0)
            self.assertAlmostEqual(k.value, u_arr_diff[i])
            self.assertIsInstance(r, relentless.model.IndependentVariable)
            self.assertIsInstance(k, relentless.model.IndependentVariable)
            if i < s.num_knots - 1:
                dvars.append(k)
        self.assertCountEqual(s.design_variables, dvars)

        # test value mode
        s = relentless.model.potential.BondSpline(
            types=("1",), num_knots=3, mode="value"
        )
        s.from_array(types=("1"), r=r_arr, u=u_arr)

        dvars = []
        for i, (r, k) in enumerate(s.knots(types=("1"))):
            self.assertAlmostEqual(r.value, 1.0)
            self.assertAlmostEqual(k.value, u_arr[i])
            self.assertIsInstance(r, relentless.model.IndependentVariable)
            self.assertIsInstance(k, relentless.model.IndependentVariable)
            if i != s.num_knots - 1:
                dvars.append(k)
        self.assertCountEqual(s.design_variables, dvars)

        # test invalid r and u shapes
        r_arr = [2, 3]
        with self.assertRaises(ValueError):
            s.from_array(types=("1"), r=r_arr, u=u_arr)

        r_arr = [1, 2, 3]
        u_arr = [1, 2]
        with self.assertRaises(ValueError):
            s.from_array(types=("1"), r=r_arr, u=u_arr)

    def test_energy(self):
        """Test energy method"""
        r_arr = [1, 2, 3]
        u_arr = [9, 4, 1]

        # test diff mode
        s = relentless.model.potential.BondSpline(types=("1",), num_knots=3)
        s.from_array(types=("1"), r=r_arr, u=u_arr)
        u_actual = numpy.array([6.25, 2.25, 1])
        u = s.energy(type_=("1"), r=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(u, u_actual)

        # test value mode
        s = relentless.model.potential.BondSpline(
            types=("1",), num_knots=3, mode="value"
        )
        s.from_array(types=("1"), r=r_arr, u=u_arr)
        u_actual = numpy.array([6.25, 2.25, 1])
        u = s.energy(type_=("1"), r=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(u, u_actual)

        # test BondSpline with 2 knots
        s = relentless.model.potential.BondSpline(
            types=("1",), num_knots=2, mode="value"
        )
        s.from_array(types=("1"), r=[1, 2], u=[4, 2])
        u = s.energy(type_=("1"), r=1.5)
        self.assertAlmostEqual(u, 3)

    def test_force(self):
        """Test force method"""
        r_arr = [1, 2, 3]
        u_arr = [9, 4, 1]

        # test diff mode
        s = relentless.model.potential.BondSpline(types=("1",), num_knots=3)
        s.from_array(types=("1"), r=r_arr, u=u_arr)
        f_actual = numpy.array([5, 3, 0])
        f = s.force(type_=("1"), r=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(f, f_actual)

        # test value mode
        s = relentless.model.potential.BondSpline(
            types=("1",), num_knots=3, mode="value"
        )
        s.from_array(types=("1"), r=r_arr, u=u_arr)
        f_actual = numpy.array([5, 3, 0])
        f = s.force(type_=("1"), r=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(f, f_actual)

        # test BondSpline with 2 knots
        s = relentless.model.potential.BondSpline(
            types=("1",), num_knots=2, mode="value"
        )
        s.from_array(types=("1"), r=[1, 2], u=[4, 2])
        f = s.force(type_=("1"), r=1.5)
        self.assertAlmostEqual(f, 2)

    def test_derivative(self):
        """Test derivative method"""
        r_arr = [1, 2, 3]
        u_arr = [9, 4, 1]

        # test diff mode
        s = relentless.model.potential.BondSpline(types=("1",), num_knots=3)
        s.from_array(types=("1"), r=r_arr, u=u_arr)
        d_actual = numpy.array([1.125, 0.625, 0])
        param = list(s.knots(("1")))[1][1]
        d = s.derivative(type_=("1"), var=param, r=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(d, d_actual)

        # test value mode
        s = relentless.model.potential.BondSpline(
            types=("1",), num_knots=3, mode="value"
        )
        s.from_array(types=("1"), r=r_arr, u=u_arr)
        d_actual = numpy.array([0.75, 0.75, 0])
        param = list(s.knots(("1")))[1][1]
        d = s.derivative(type_=("1"), var=param, r=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(d, d_actual)

    def test_json(self):
        r_arr = [1, 2, 3]
        u_arr = [9, 4, 1]
        u_arr_diff = [5, 3, 1]

        s = relentless.model.potential.BondSpline(types=("1",), num_knots=3)
        s.from_array(types=("1"), r=r_arr, u=u_arr)
        data = s.to_json()

        s2 = relentless.model.potential.BondSpline.from_json(data)
        for i, (r, k) in enumerate(s2.knots(types=("1"))):
            self.assertAlmostEqual(r.value, 1.0)
            self.assertAlmostEqual(k.value, u_arr_diff[i])


if __name__ == "__main__":
    unittest.main()
