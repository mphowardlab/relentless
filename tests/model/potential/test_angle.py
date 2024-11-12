"""Unit tests for angle module."""

import tempfile
import unittest

import numpy

import relentless


class test_AngleParameters(unittest.TestCase):
    """Unit tests for relentless.angle.AngleParameters"""

    def test_init(self):
        """Test creation from data"""
        types = ("A", "B")
        params = ("energy", "mass")

        # test construction with tuple input
        m = relentless.model.potential.angle.AngleParameters(
            types=("A", "B"), params=("energy", "mass")
        )
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)

        # test construction with list input
        m = relentless.model.potential.angle.AngleParameters(
            types=["A", "B"], params=("energy", "mass")
        )
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)

        # test construction with mixed tuple/list input
        m = relentless.model.potential.angle.AngleParameters(
            types=("A", "B"), params=["energy", "mass"]
        )
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)

        # test construction with int type parameters
        with self.assertRaises(TypeError):
            m = relentless.model.potential.angle.AngleParameters(
                types=("A", "B"), params=(1, 2)
            )

        # test construction with mixed type parameters
        with self.assertRaises(TypeError):
            m = relentless.model.potential.angle.AngleParameters(
                types=("A", "B"), params=("1", 2)
            )

    def test_param_types(self):
        """Test various get and set methods on angle parameter types"""
        m = relentless.model.potential.angle.AngleParameters(
            types=("A", "B"), params=("energy", "mass")
        )

        self.assertEqual(m["A"]["energy"], None)
        self.assertEqual(m["A"]["mass"], None)
        self.assertEqual(m["B"]["energy"], None)
        self.assertEqual(m["B"]["mass"], None)

        # test setting per-type params
        m["A"].update(energy=1.5, mass=2.5)
        m["B"].update(energy=0.5, mass=0.7)

        self.assertEqual(m["A"]["energy"], 1.5)
        self.assertEqual(m["A"]["mass"], 2.5)
        self.assertEqual(m["B"]["energy"], 0.5)
        self.assertEqual(m["B"]["mass"], 0.7)


class LinPot(relentless.model.potential.angle.AnglePotential):
    """Linear angle potential function"""

    def __init__(self, types, params):
        super().__init__(types, params)

    def to_json(self):
        data = super().to_json()
        data["params"] = self.coeff.params
        return data

    def energy(self, types, theta):
        m = self.coeff[types]["m"]

        theta, u, s = self._zeros(theta)
        u[:] = m * theta
        if s:
            u = u.item()
        return u

    def force(self, types, theta):
        m = self.coeff[types]["m"]

        theta, f, s = self._zeros(theta)
        f[:] = -m
        if s:
            f = f.item()
        return f

    def derivative(self, types, param, theta):
        theta, d, s = self._zeros(theta)
        if param == "m":
            d[:] = theta
        if s:
            d = d.item()
        return d


class TwoVarPot(relentless.model.potential.angle.AnglePotential):
    """Mock potential function"""

    def __init__(self, types, params):
        super().__init__(types, params)

    @classmethod
    def from_json(cls, data):
        raise NotImplementedError()

    def energy(self, theta, x, y, **params):
        pass

    def force(self, theta, x, y, **params):
        pass

    def derivative(self, param, theta, **params):
        # not real derivative, just used to test functionality
        theta, d, s = self._zeros(theta)
        if param == "x":
            d[:] = 2 * theta
        elif param == "y":
            d[:] = 3 * theta
        if s:
            d = d.item()
        return d


class test_AnglePotential(unittest.TestCase):
    """Unit tests for relentless.model.angle.AnglePotential"""

    def test_init(self):
        """Test creation from data"""
        # test creation with only m
        p = LinPot(types=("1",), params=("m",))
        p.coeff["1"]["m"] = 3.5

        coeff = relentless.model.potential.angle.AngleParameters(
            types=("1",), params=("m")
        )
        coeff["1"]["m"] = 3.5

        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertEqual(p.coeff.evaluate(("1")), coeff.evaluate(("1")))

    def test_energy(self):
        """Test energy method"""
        p = LinPot(types=("1",), params=("m",))
        p.coeff["1"]["m"] = 2.0

        # test with no cutoffs
        u = p.energy(types=("1"), theta=0.5)
        self.assertAlmostEqual(u, 1.0)
        u = p.energy(types=("1"), theta=[0.25, 0.75])
        numpy.testing.assert_allclose(u, [0.5, 1.5])

    def test_force(self):
        """Test force method"""
        p = LinPot(types=("1",), params=("m",))
        p.coeff["1"]["m"] = 2.0

        # test with no cutoffs
        f = p.force(types=("1"), theta=0.5)
        self.assertAlmostEqual(f, -2.0)
        f = p.force(types=("1"), theta=[0.25, 0.75])
        numpy.testing.assert_allclose(f, [-2.0, -2.0])

    def test_derivative_values(self):
        """Test derivative method with different param values"""
        p = LinPot(types=("1",), params=("m",))
        x = relentless.model.IndependentVariable(value=2.0)
        p.coeff["1"]["m"] = x

        # test with no cutoffs
        d = p.derivative(types=("1"), param=x, theta=0.5)
        self.assertAlmostEqual(d, 0.5)
        d = p.derivative(types=("1"), param=x, theta=[0.25, 0.75])
        numpy.testing.assert_allclose(d, [0.25, 0.75])

    def test_derivative_types(self):
        """Test derivative method with different param types."""
        q = LinPot(types=("1",), params=("m",))
        x = relentless.model.IndependentVariable(value=4.0)
        y = relentless.model.IndependentVariable(value=64.0)
        z = relentless.model.GeometricMean(x, y)
        q.coeff["1"]["m"] = z

        # test with respect to dependent variable parameter
        d = q.derivative(types=("1"), var=z, theta=2.0)
        self.assertAlmostEqual(d, 2.0)

        # test with respect to independent variable on which parameter is dependent
        d = q.derivative(types=("1"), var=x, theta=1.5)
        self.assertAlmostEqual(d, 3.0)
        d = q.derivative(types=("1"), var=y, theta=4.0)
        self.assertAlmostEqual(d, 0.5)

        # test invalid derivative w.r.t. scalar
        a = 2.5
        q.coeff["1"]["m"] = a
        with self.assertRaises(TypeError):
            d = q.derivative(types=("1"), var=a, theta=2.0)

        # test with respect to independent variable which is
        # related to a SameAs variable
        r = TwoVarPot(types=("1",), params=("x", "y"))

        r.coeff["1"]["x"] = x
        r.coeff["1"]["y"] = relentless.model.variable.SameAs(x)
        d = r.derivative(types=("1"), var=x, theta=4.0)
        self.assertAlmostEqual(d, 20.0)

        r.coeff["1"]["y"] = x
        r.coeff["1"]["x"] = relentless.model.variable.SameAs(x)
        d = r.derivative(types=("1"), var=x, theta=4.0)
        self.assertAlmostEqual(d, 20.0)

    def test_iteration(self):
        """Test iteration on typesPotential object"""
        p = LinPot(types=("1",), params=("m",))
        for types in p.coeff:
            p.coeff[types]["m"] = 2.0

        self.assertEqual(dict(p.coeff["1"]), {"m": 2.0})

    def test_json(self):
        """Test saving to file"""
        p = LinPot(types=("1",), params=("m"))
        p.coeff["1"]["m"] = 2.0

        data = p.to_json()
        self.assertEqual(data["id"], p.id)
        self.assertEqual(data["name"], p.name)

        p2 = LinPot.from_json(data)
        self.assertEqual(p2.coeff["1"]["m"], 2.0)

    def test_save(self):
        """Test saving to file"""
        temp = tempfile.NamedTemporaryFile()
        p = LinPot(types=("1",), params=("m"))
        p.coeff["1"]["m"] = 2.0
        p.save(temp.name)

        p2 = LinPot.from_file(temp.name)
        self.assertEqual(p2.coeff["1"]["m"], 2.0)

        temp.close()


class test_HarmonicAngle(unittest.TestCase):
    """Unit tests for relentless.model.potential.HarmonicAngle"""

    def test_init(self):
        """Test creation from data"""
        harmonic_angle = relentless.model.potential.HarmonicAngle(types=("1",))
        coeff = relentless.model.potential.AngleParameters(
            types=("1",),
            params=(
                "k",
                "theta0",
            ),
        )
        self.assertCountEqual(harmonic_angle.coeff.types, coeff.types)
        self.assertCountEqual(harmonic_angle.coeff.params, coeff.params)

    def test_energy(self):
        """Test _energy method"""
        harmonic_angle = relentless.model.potential.HarmonicAngle(types=("1",))
        harmonic_angle.coeff["1"].update(k=1000, theta0=1.0)
        # test scalar r
        theta_input = 0.5
        u_actual = 125.0
        u = harmonic_angle.energy(types=("1"), theta=theta_input)
        self.assertAlmostEqual(u, u_actual)

        # test array r
        theta_input = numpy.array([0.0, 0.5, 1.0])
        u_actual = numpy.array([500.0, 125.0, 0.0])
        u = harmonic_angle.energy(types=("1"), theta=theta_input)
        numpy.testing.assert_allclose(u, u_actual)

    def test_force(self):
        """Test _force method"""
        harmonic_angle = relentless.model.potential.HarmonicAngle(types=("1",))
        harmonic_angle.coeff["1"].update(k=1000, theta0=1.0)

        # test scalar r
        theta_input = 0.5
        f_actual = 500
        f = harmonic_angle.force(types=("1"), theta=theta_input)
        self.assertAlmostEqual(f, f_actual)

        # test array r
        theta_input = numpy.array([0.0, 0.5, 1.0])
        f_actual = numpy.array([1000, 500, 0])
        f = harmonic_angle.force(types=("1"), theta=theta_input)
        numpy.testing.assert_allclose(f, f_actual)

    def test_derivative(self):
        """Test _derivative method"""
        harmonic_angle = relentless.model.potential.HarmonicAngle(types=("1",))

        # w.r.t. k
        # test scalar r
        theta_input = 0.5
        d_actual = 0
        d = harmonic_angle._derivative(param="k", theta=theta_input)
        self.assertAlmostEqual(d, d_actual)

        # test array r
        theta_input = numpy.array([0, 1, 1.5])
        d_actual = numpy.array([numpy.inf, -0.061523438, -0.0054794417])
        d = harmonic_angle._derivative(
            param="epsilon", theta=theta_input, epsilon=1.0, sigma=0.5
        )
        numpy.testing.assert_allclose(d, d_actual)

        # w.r.t. sigma
        # test scalar r
        theta_input = 0.5
        d_actual = 48
        d = harmonic_angle._derivative(
            param="sigma", theta=theta_input, epsilon=1.0, sigma=0.5
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        theta_input = numpy.array([0, 1, 1.5])
        d_actual = numpy.array([numpy.inf, -0.7265625, -0.06566298])
        d = harmonic_angle._derivative(
            param="sigma", theta=theta_input, epsilon=1.0, sigma=0.5
        )
        numpy.testing.assert_allclose(d, d_actual)

        # test invalid param
        with self.assertRaises(ValueError):
            harmonic_angle._derivative(
                param="simga", theta=theta_input, epsilon=1.0, sigma=1.0
            )

    def test_json(self):
        harmonic_angle = relentless.model.potential.HarmonicAngle(types=("1",))
        harmonic_angle.coeff["1"].update(k=1000, theta0=1.0)
        data = harmonic_angle.to_json()

        harmonic_angle2 = relentless.model.potential.HarmonicAngle.from_json(data)
        self.assertEqual(harmonic_angle2.coeff["1"]["k"], 1000)
        self.assertEqual(harmonic_angle2.coeff["1"]["theta0"], 1.0)


class test_CosineSquaredAngle(unittest.TestCase):
    """Unit tests for relentless.model.potential.CosineSquaredAngle"""

    def test_init(self):
        """Test creation from data"""
        cosine_squred_angle = relentless.model.potential.CosineSquaredAngle(
            types=("1",)
        )
        coeff = relentless.model.potential.AngleParameters(
            types=("1",),
            params=(
                "k",
                "theta0",
            ),
        )
        self.assertCountEqual(cosine_squred_angle.coeff.types, coeff.types)
        self.assertCountEqual(cosine_squred_angle.coeff.params, coeff.params)

    def test_energy(self):
        """Test _energy method"""
        cosine_squred_angle = relentless.model.potential.CosineSquaredAngle(
            types=("1",)
        )
        cosine_squred_angle.coeff["1"].update(k=1000, theta0=numpy.pi / 2)
        # test scalar r
        theta_input = numpy.pi
        u_actual = 1000
        u = cosine_squred_angle.energy(types=("1"), theta=theta_input)
        self.assertAlmostEqual(u, u_actual)

        # test array r
        theta_input = numpy.array([0.0, numpy.pi / 2, numpy.pi])
        u_actual = numpy.array([1000.0, 0, 1000.0])
        u = cosine_squred_angle.energy(types=("1"), theta=theta_input)
        numpy.testing.assert_allclose(u, u_actual)

    def test_force(self):
        """Test _force method"""
        cosine_squred_angle = relentless.model.potential.CosineSquaredAngle(
            types=("1",)
        )
        cosine_squred_angle.coeff["1"].update(k=1000, theta0=numpy.pi / 2)

        # test scalar r
        theta_input = numpy.pi
        f_actual = 0
        f = cosine_squred_angle.force(types=("1"), theta=theta_input)
        self.assertAlmostEqual(f, f_actual)

        # test array r
        theta_input = numpy.array([numpy.pi / 4, numpy.pi / 2, 3 * numpy.pi / 4])
        f_actual = numpy.array([1000, 0, -1000])
        f = cosine_squred_angle.force(types=("1"), theta=theta_input)
        numpy.testing.assert_allclose(f, f_actual)

    def test_derivative(self):
        """Test _derivative method"""
        cosine_squred_angle = relentless.model.potential.CosineSquaredAngle(
            types=("1",)
        )

        # w.r.t. k
        # test scalar r
        theta_input = 0.5
        d_actual = 0
        d = cosine_squred_angle._derivative(param="k", theta=theta_input)
        self.assertAlmostEqual(d, d_actual)

        # test array r
        theta_input = numpy.array([0, 1, 1.5])
        d_actual = numpy.array([numpy.inf, -0.061523438, -0.0054794417])
        d = cosine_squred_angle._derivative(
            param="epsilon", theta=theta_input, epsilon=1.0, sigma=0.5
        )
        numpy.testing.assert_allclose(d, d_actual)

        # w.r.t. sigma
        # test scalar r
        theta_input = 0.5
        d_actual = 48
        d = cosine_squred_angle._derivative(
            param="sigma", theta=theta_input, epsilon=1.0, sigma=0.5
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        theta_input = numpy.array([0, 1, 1.5])
        d_actual = numpy.array([numpy.inf, -0.7265625, -0.06566298])
        d = cosine_squred_angle._derivative(
            param="sigma", theta=theta_input, epsilon=1.0, sigma=0.5
        )
        numpy.testing.assert_allclose(d, d_actual)

        # test invalid param
        with self.assertRaises(ValueError):
            cosine_squred_angle._derivative(
                param="simga", theta=theta_input, epsilon=1.0, sigma=1.0
            )

    def test_json(self):
        cosine_squred_angle = relentless.model.potential.CosineSquaredAngle(
            types=("1",)
        )
        cosine_squred_angle.coeff["1"].update(k=1000, theta0=1.0)
        data = cosine_squred_angle.to_json()

        cosine_squred_angle2 = relentless.model.potential.CosineSquaredAngle.from_json(
            data
        )
        self.assertEqual(cosine_squred_angle2.coeff["1"]["k"], 1000)
        self.assertEqual(cosine_squred_angle2.coeff["1"]["theta0"], 1.0)


class test_AngleSpline(unittest.TestCase):
    """Unit tests for relentless.model.potential.AngleSpline"""

    def test_init(self):
        """Test creation from data"""
        # test diff mode
        s = relentless.model.potential.AngleSpline(types=("1",), num_knots=3)
        self.assertEqual(s.num_knots, 3)
        self.assertEqual(s.mode, "diff")
        coeff = relentless.model.potential.AngleParameters(
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
        s = relentless.model.potential.AngleSpline(
            types=("1",), num_knots=3, mode="value"
        )
        self.assertEqual(s.num_knots, 3)
        self.assertEqual(s.mode, "value")
        coeff = relentless.model.potential.AngleParameters(
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
            s = relentless.model.potential.AngleSpline(types=("1",), num_knots=1)

        # test invalid mode
        with self.assertRaises(ValueError):
            s = relentless.model.potential.AngleSpline(
                types=("1",), num_knots=3, mode="val"
            )

    def test_from_array(self):
        """Test from_array method and knots generator"""
        r_arr = [1, 2, 3]
        u_arr = [9, 4, 1]
        u_arr_diff = [5, 3, 1]

        # test diff mode
        s = relentless.model.potential.AngleSpline(types=("1",), num_knots=3)
        s.from_array(types=("1"), theta=r_arr, u=u_arr)

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
        s = relentless.model.potential.AngleSpline(
            types=("1",), num_knots=3, mode="value"
        )
        s.from_array(types=("1"), theta=r_arr, u=u_arr)

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
            s.from_array(types=("1"), theta=r_arr, u=u_arr)

        r_arr = [1, 2, 3]
        u_arr = [1, 2]
        with self.assertRaises(ValueError):
            s.from_array(types=("1"), theta=r_arr, u=u_arr)

    def test_energy(self):
        """Test energy method"""
        r_arr = [1, 2, 3]
        u_arr = [9, 4, 1]

        # test diff mode
        s = relentless.model.potential.AngleSpline(types=("1",), num_knots=3)
        s.from_array(types=("1"), theta=r_arr, u=u_arr)
        u_actual = numpy.array([6.25, 2.25, 1])
        u = s.energy(types=("1"), theta=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(u, u_actual)

        # test value mode
        s = relentless.model.potential.AngleSpline(
            types=("1",), num_knots=3, mode="value"
        )
        s.from_array(types=("1"), theta=r_arr, u=u_arr)
        u_actual = numpy.array([6.25, 2.25, 1])
        u = s.energy(types=("1"), theta=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(u, u_actual)

        # test AngleSpline with 2 knots
        s = relentless.model.potential.AngleSpline(
            types=("1",), num_knots=2, mode="value"
        )
        s.from_array(types=("1"), theta=[1, 2], u=[4, 2])
        u = s.energy(types=("1"), theta=1.5)
        self.assertAlmostEqual(u, 3)

    def test_force(self):
        """Test force method"""
        r_arr = [1, 2, 3]
        u_arr = [9, 4, 1]

        # test diff mode
        s = relentless.model.potential.AngleSpline(types=("1",), num_knots=3)
        s.from_array(types=("1"), theta=r_arr, u=u_arr)
        f_actual = numpy.array([5, 3, 0])
        f = s.force(types=("1"), theta=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(f, f_actual)

        # test value mode
        s = relentless.model.potential.AngleSpline(
            types=("1",), num_knots=3, mode="value"
        )
        s.from_array(types=("1"), theta=r_arr, u=u_arr)
        f_actual = numpy.array([5, 3, 0])
        f = s.force(types=("1"), theta=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(f, f_actual)

        # test AngleSpline with 2 knots
        s = relentless.model.potential.AngleSpline(
            types=("1",), num_knots=2, mode="value"
        )
        s.from_array(types=("1"), theta=[1, 2], u=[4, 2])
        f = s.force(types=("1"), theta=1.5)
        self.assertAlmostEqual(f, 2)

    def test_derivative(self):
        """Test derivative method"""
        r_arr = [1, 2, 3]
        u_arr = [9, 4, 1]

        # test diff mode
        s = relentless.model.potential.AngleSpline(types=("1",), num_knots=3)
        s.from_array(types=("1"), theta=r_arr, u=u_arr)
        d_actual = numpy.array([1.125, 0.625, 0])
        param = list(s.knots(("1")))[1][1]
        d = s.derivative(types=("1"), var=param, r=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(d, d_actual)

        # test value mode
        s = relentless.model.potential.AngleSpline(
            types=("1",), num_knots=3, mode="value"
        )
        s.from_array(types=("1"), theta=r_arr, u=u_arr)
        d_actual = numpy.array([0.75, 0.75, 0])
        param = list(s.knots(("1")))[1][1]
        d = s.derivative(types=("1"), var=param, r=[1.5, 2.5, 3.5])
        numpy.testing.assert_allclose(d, d_actual)

    def test_json(self):
        r_arr = [1, 2, 3]
        u_arr = [9, 4, 1]
        u_arr_diff = [5, 3, 1]

        s = relentless.model.potential.AngleSpline(types=("1",), num_knots=3)
        s.from_array(types=("1"), theta=r_arr, u=u_arr)
        data = s.to_json()

        s2 = relentless.model.potential.AngleSpline.from_json(data)
        for i, (r, k) in enumerate(s2.knots(types=("1"))):
            self.assertAlmostEqual(r.value, 1.0)
            self.assertAlmostEqual(k.value, u_arr_diff[i])


if __name__ == "__main__":
    unittest.main()
