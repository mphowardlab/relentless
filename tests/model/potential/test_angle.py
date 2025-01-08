"""Unit tests for angle module."""

import tempfile
import unittest

import numpy

import relentless


class LinPotAngle(relentless.model.potential.angle.AnglePotential):
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

    def _derivative(self, param, theta, **params):
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

    def _derivative(self, param, theta, **params):
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
        p = LinPotAngle(types=("1",), params=("m",))
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
        p = LinPotAngle(types=("1",), params=("m",))
        p.coeff["1"]["m"] = 2.0

        # test with no cutoffs
        u = p.energy(types=("1"), theta=0.5)
        self.assertAlmostEqual(u, 1.0)
        u = p.energy(types=("1"), theta=[0.25, 0.75])
        numpy.testing.assert_allclose(u, [0.5, 1.5])

    def test_force(self):
        """Test force method"""
        p = LinPotAngle(types=("1",), params=("m",))
        p.coeff["1"]["m"] = 2.0

        # test with no cutoffs
        f = p.force(types=("1"), theta=0.5)
        self.assertAlmostEqual(f, -2.0)
        f = p.force(types=("1"), theta=[0.25, 0.75])
        numpy.testing.assert_allclose(f, [-2.0, -2.0])

    def test_derivative_values(self):
        """Test derivative method with different param values"""
        p = LinPotAngle(types=("1",), params=("m",))
        x = relentless.model.IndependentVariable(value=2.0)
        p.coeff["1"]["m"] = x

        # test with no cutoffs
        d = p.derivative(type_=("1"), var=x, r=0.5)
        self.assertAlmostEqual(d, 0.5)
        d = p.derivative(type_=("1"), var=x, r=[0.25, 0.75])
        numpy.testing.assert_allclose(d, [0.25, 0.75])

    def test_derivative_types(self):
        """Test derivative method with different param types."""
        q = LinPotAngle(types=("1",), params=("m",))
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
        p = LinPotAngle(types=("1",), params=("m",))
        for types in p.coeff:
            p.coeff[types]["m"] = 2.0

        self.assertEqual(dict(p.coeff["1"]), {"m": 2.0})

    def test_json(self):
        """Test saving to file"""
        p = LinPotAngle(types=("1",), params=("m"))
        p.coeff["1"]["m"] = 2.0

        data = p.to_json()
        self.assertEqual(data["id"], p.id)
        self.assertEqual(data["name"], p.name)

        p2 = LinPotAngle.from_json(data)
        self.assertEqual(p2.coeff["1"]["m"], 2.0)

    def test_save(self):
        """Test saving to file"""
        temp = tempfile.NamedTemporaryFile()
        p = LinPotAngle(types=("1",), params=("m"))
        p.coeff["1"]["m"] = 2.0
        p.save(temp.name)

        p2 = LinPotAngle.from_file(temp.name)
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
        u = harmonic_angle.energy(type_=("1"), theta=theta_input)
        self.assertAlmostEqual(u, u_actual)

        # test array r
        theta_input = numpy.array([0.0, 0.5, 1.0])
        u_actual = numpy.array([500.0, 125.0, 0.0])
        u = harmonic_angle.energy(type_=("1"), theta=theta_input)
        numpy.testing.assert_allclose(u, u_actual)

    def test_force(self):
        """Test _force method"""
        harmonic_angle = relentless.model.potential.HarmonicAngle(types=("1",))
        harmonic_angle.coeff["1"].update(k=1000, theta0=1.0)

        # test scalar r
        theta_input = 0.5
        f_actual = 500
        f = harmonic_angle.force(type_=("1"), theta=theta_input)
        self.assertAlmostEqual(f, f_actual)

        # test array r
        theta_input = numpy.array([0.0, 0.5, 1.0])
        f_actual = numpy.array([1000, 500, 0])
        f = harmonic_angle.force(type_=("1"), theta=theta_input)
        numpy.testing.assert_allclose(f, f_actual)

    def test_derivative(self):
        """Test _derivative method"""
        harmonic_angle = relentless.model.potential.HarmonicAngle(types=("1",))

        # w.r.t. k
        # test scalar r
        theta_input = 0.5
        d_actual = 0.125
        d = harmonic_angle._derivative(param="k", theta=theta_input, k=1000, theta0=1.0)
        self.assertAlmostEqual(d, d_actual)

        # test array r
        theta_input = numpy.array([0, 1, 1.5])
        d_actual = numpy.array([0.50, 0.0, 0.125])
        d = harmonic_angle._derivative(param="k", theta=theta_input, k=1000, theta0=1.0)
        numpy.testing.assert_allclose(d, d_actual)

        # w.r.t. theta0
        # test scalar r
        theta_input = 0.5
        d_actual = 500
        d = harmonic_angle._derivative(
            param="theta0", theta=theta_input, k=1000.0, theta0=1.0
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        theta_input = numpy.array([0, 1, 1.5])
        d_actual = numpy.array([1000, 0, -500])
        d = harmonic_angle._derivative(
            param="theta0", theta=theta_input, k=1000.0, theta0=1.0
        )
        numpy.testing.assert_allclose(d, d_actual)

        # test invalid param
        with self.assertRaises(ValueError):
            harmonic_angle._derivative(
                param="ro", theta=theta_input, k=1000.0, theta0=1.0
            )

    def test_json(self):
        harmonic_angle = relentless.model.potential.HarmonicAngle(types=("1",))
        harmonic_angle.coeff["1"].update(k=1000, theta0=1.0)
        data = harmonic_angle.to_json()

        harmonic_angle2 = relentless.model.potential.HarmonicAngle.from_json(data)
        self.assertEqual(harmonic_angle2.coeff["1"]["k"], 1000)
        self.assertEqual(harmonic_angle2.coeff["1"]["theta0"], 1.0)


class test_CosineAngle(unittest.TestCase):
    """Unit tests for relentless.model.potential.CosineAngle"""

    def test_init(self):
        """Test creation from data"""
        cosine_angle = relentless.model.potential.CosineAngle(types=("1",))
        coeff = relentless.model.potential.AngleParameters(
            types=("1",),
            params=("k",),
        )
        self.assertCountEqual(cosine_angle.coeff.types, coeff.types)
        self.assertCountEqual(cosine_angle.coeff.params, coeff.params)

    def test_energy(self):
        """Test _energy method"""
        cosine_angle = relentless.model.potential.CosineAngle(types=("1",))
        cosine_angle.coeff["1"].update(
            k=1000,
        )
        # test scalar r
        theta_input = numpy.pi / 2
        u_actual = 1000
        u = cosine_angle.energy(type_=("1"), theta=theta_input)
        self.assertAlmostEqual(u, u_actual)

        # test array r
        theta_input = numpy.array([0.0, numpy.pi / 4, numpy.pi])
        u_actual = numpy.array([2000.0, 1707.10678119, 0])
        u = cosine_angle.energy(type_=("1"), theta=theta_input)
        numpy.testing.assert_allclose(u, u_actual)

    def test_force(self):
        """Test _force method"""
        cosine_angle = relentless.model.potential.CosineAngle(types=("1",))
        cosine_angle.coeff["1"].update(k=1000)

        # test scalar r
        theta_input = numpy.pi / 2
        f_actual = 1000
        f = cosine_angle.force(type_=("1"), theta=theta_input)
        self.assertAlmostEqual(f, f_actual)

        # test array r
        theta_input = numpy.array([0.0, numpy.pi / 4, numpy.pi])
        f_actual = numpy.array([0.0, 707.106781187, 0])
        f = cosine_angle.force(type_=("1"), theta=theta_input)
        numpy.testing.assert_allclose(f, f_actual, atol=1e-10)

    def test_derivative(self):
        """Test _derivative method"""
        cosine_angle = relentless.model.potential.CosineAngle(types=("1",))

        # w.r.t. k
        # test scalar r
        theta_input = numpy.pi
        d_actual = 0.0
        d = cosine_angle._derivative(param="k", theta=theta_input, k=1000)
        self.assertAlmostEqual(d, d_actual)

        # test array r
        theta_input = numpy.array([0.0, numpy.pi / 4, numpy.pi])
        d_actual = numpy.array([2.0, 1.70710678119, 0])
        d = cosine_angle._derivative(param="k", theta=theta_input, k=1000)
        numpy.testing.assert_allclose(d, d_actual)

        # test invalid param
        with self.assertRaises(ValueError):
            cosine_angle._derivative(
                param="thetao",
                theta=theta_input,
                k=1000.0,
            )

    def test_json(self):
        cosine_angle = relentless.model.potential.CosineAngle(types=("1",))
        cosine_angle.coeff["1"].update(k=1000)
        data = cosine_angle.to_json()

        cosine_angle2 = relentless.model.potential.CosineAngle.from_json(data)
        self.assertEqual(cosine_angle2.coeff["1"]["k"], 1000)


class test_HarmonicCosineAngle(unittest.TestCase):
    """Unit tests for relentless.model.potential.HarmonicCosineAngle"""

    def test_init(self):
        """Test creation from data"""
        cosine_squred_angle = relentless.model.potential.HarmonicCosineAngle(
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
        cosine_squred_angle = relentless.model.potential.HarmonicCosineAngle(
            types=("1",)
        )
        cosine_squred_angle.coeff["1"].update(k=1000, theta0=numpy.pi / 2)
        # test scalar r
        theta_input = numpy.pi
        u_actual = 1000
        u = cosine_squred_angle.energy(type_=("1"), theta=theta_input)
        self.assertAlmostEqual(u, u_actual)

        # test array r
        theta_input = numpy.array([0.0, numpy.pi / 2, numpy.pi])
        u_actual = numpy.array([1000.0, 0, 1000.0])
        u = cosine_squred_angle.energy(type_=("1"), theta=theta_input)
        numpy.testing.assert_allclose(u, u_actual)

    def test_force(self):
        """Test _force method"""
        cosine_squred_angle = relentless.model.potential.HarmonicCosineAngle(
            types=("1",)
        )
        cosine_squred_angle.coeff["1"].update(k=1000, theta0=numpy.pi / 2)

        # test scalar r
        theta_input = numpy.pi
        f_actual = 0
        f = cosine_squred_angle.force(type_=("1"), theta=theta_input)
        self.assertAlmostEqual(f, f_actual)

        # test array r
        theta_input = numpy.array([numpy.pi / 4, numpy.pi / 2, 3 * numpy.pi / 4])
        f_actual = numpy.array([1000, 0, -1000])
        f = cosine_squred_angle.force(type_=("1"), theta=theta_input)
        numpy.testing.assert_allclose(f, f_actual)

    def test_derivative(self):
        """Test _derivative method"""
        harmonic_cosine_angle = relentless.model.potential.HarmonicCosineAngle(
            types=("1",)
        )

        # w.r.t. k
        # test scalar r
        theta_input = numpy.pi
        d_actual = 1
        d = harmonic_cosine_angle._derivative(
            param="k", theta=theta_input, k=1000, theta0=numpy.pi / 2
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        theta_input = numpy.array([numpy.pi / 4, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([0.50, 0.0, 1])
        d = harmonic_cosine_angle._derivative(
            param="k", theta=theta_input, k=1000, theta0=numpy.pi / 2
        )
        numpy.testing.assert_allclose(d, d_actual)

        # w.r.t. theta0
        # test scalar r
        theta_input = numpy.pi
        d_actual = -2000
        d = harmonic_cosine_angle._derivative(
            param="theta0", theta=theta_input, k=1000.0, theta0=numpy.pi / 2
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        theta_input = numpy.array([numpy.pi / 4, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([1414.21356237, 0, -2000])
        d = harmonic_cosine_angle._derivative(
            param="theta0", theta=theta_input, k=1000.0, theta0=numpy.pi / 2
        )
        numpy.testing.assert_allclose(d, d_actual)

        # test invalid param
        with self.assertRaises(ValueError):
            harmonic_cosine_angle._derivative(
                param="thetao", theta=theta_input, k=1000.0, theta0=1.0
            )

    def test_json(self):
        cosine_squred_angle = relentless.model.potential.HarmonicCosineAngle(
            types=("1",)
        )
        cosine_squred_angle.coeff["1"].update(k=1000, theta0=1.0)
        data = cosine_squred_angle.to_json()

        cosine_squred_angle2 = relentless.model.potential.HarmonicCosineAngle.from_json(
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


if __name__ == "__main__":
    unittest.main()
