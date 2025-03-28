"""Unit tests for dihedral module."""

import tempfile
import unittest

import numpy

import relentless


class test_DihedralParameters(unittest.TestCase):
    """Unit tests for relentless.dihedral.DihedralParameters"""

    def test_init(self):
        """Test creation from data"""
        types = ("A", "B")
        params = ("energy", "mass")

        # test construction with tuple input
        m = relentless.model.potential.dihedral.DihedralParameters(
            types=("A", "B"), params=("energy", "mass")
        )
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)

        # test construction with list input
        m = relentless.model.potential.dihedral.DihedralParameters(
            types=["A", "B"], params=("energy", "mass")
        )
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)

        # test construction with mixed tuple/list input
        m = relentless.model.potential.dihedral.DihedralParameters(
            types=("A", "B"), params=["energy", "mass"]
        )
        self.assertEqual(m.types, types)
        self.assertEqual(m.params, params)

        # test construction with int type parameters
        with self.assertRaises(TypeError):
            m = relentless.model.potential.dihedral.DihedralParameters(
                types=("A", "B"), params=(1, 2)
            )

        # test construction with mixed type parameters
        with self.assertRaises(TypeError):
            m = relentless.model.potential.dihedral.DihedralParameters(
                types=("A", "B"), params=("1", 2)
            )

    def test_param_types(self):
        """Test various get and set methods on dihedral parameter types"""
        m = relentless.model.potential.dihedral.DihedralParameters(
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


class LinPotDihedral(relentless.model.potential.dihedral.DihedralPotential):
    """Linear dihedral potential function"""

    def __init__(self, types, params):
        super().__init__(types, params)

    def to_json(self):
        data = super().to_json()
        data["params"] = self.coeff.params
        return data

    def energy(self, types, phi):
        m = self.coeff[types]["m"]

        phi, u, s = self._zeros(phi)
        u[:] = m * phi
        if s:
            u = u.item()
        return u

    def force(self, types, phi):
        m = self.coeff[types]["m"]

        phi, f, s = self._zeros(phi)
        f[:] = -m
        if s:
            f = f.item()
        return f

    def _derivative(self, param, phi, **params):
        phi, d, s = self._zeros(phi)
        if param == "m":
            d[:] = phi
        if s:
            d = d.item()
        return d


class TwoVarPot(relentless.model.potential.dihedral.DihedralPotential):
    """Mock potential function"""

    def __init__(self, types, params):
        super().__init__(types, params)

    @classmethod
    def from_json(cls, data):
        raise NotImplementedError()

    def energy(self, phi, x, y, **params):
        pass

    def force(self, phi, x, y, **params):
        pass

    def _derivative(self, param, phi, **params):
        # not real derivative, just used to test functionality
        phi, d, s = self._zeros(phi)
        if param == "x":
            d[:] = 2 * phi
        elif param == "y":
            d[:] = 3 * phi
        if s:
            d = d.item()
        return d


class test_DihedralPotential(unittest.TestCase):
    """Unit tests for relentless.model.dihedral.DihedralPotential"""

    def test_init(self):
        """Test creation from data"""
        # test creation with only m
        p = LinPotDihedral(types=("1",), params=("m",))
        p.coeff["1"]["m"] = 3.5

        coeff = relentless.model.potential.dihedral.DihedralParameters(
            types=("1",), params=("m")
        )
        coeff["1"]["m"] = 3.5

        self.assertCountEqual(p.coeff.types, coeff.types)
        self.assertCountEqual(p.coeff.params, coeff.params)
        self.assertEqual(p.coeff.evaluate(("1")), coeff.evaluate(("1")))

    def test_energy(self):
        """Test energy method"""
        p = LinPotDihedral(types=("1",), params=("m",))
        p.coeff["1"]["m"] = 2.0

        # test with no cutoffs
        u = p.energy(types=("1"), phi=0.5)
        self.assertAlmostEqual(u, 1.0)
        u = p.energy(types=("1"), phi=[0.25, 0.75])
        numpy.testing.assert_allclose(u, [0.5, 1.5])

    def test_force(self):
        """Test force method"""
        p = LinPotDihedral(types=("1",), params=("m",))
        p.coeff["1"]["m"] = 2.0

        # test with no cutoffs
        f = p.force(types=("1"), phi=0.5)
        self.assertAlmostEqual(f, -2.0)
        f = p.force(types=("1"), phi=[0.25, 0.75])
        numpy.testing.assert_allclose(f, [-2.0, -2.0])

    def test_derivative_values(self):
        """Test derivative method with different param values"""
        p = LinPotDihedral(types=("1",), params=("m",))
        x = relentless.model.IndependentVariable(value=2.0)
        p.coeff["1"]["m"] = x

        # test with no cutoffs
        d = p.derivative(type_=("1"), var=x, phi=0.5)
        self.assertAlmostEqual(d, 0.5)
        d = p.derivative(type_=("1"), var=x, phi=[0.25, 0.75])
        numpy.testing.assert_allclose(d, [0.25, 0.75])

    def test_derivative_types(self):
        """Test derivative method with different param types."""
        q = LinPotDihedral(types=("1",), params=("m",))
        x = relentless.model.IndependentVariable(value=4.0)
        y = relentless.model.IndependentVariable(value=64.0)
        z = relentless.model.GeometricMean(x, y)
        q.coeff["1"]["m"] = z

        # test with respect to dependent variable parameter
        d = q.derivative(type_=("1"), var=z, phi=2.0)
        self.assertAlmostEqual(d, 2.0)

        # test with respect to independent variable on which parameter is dependent
        d = q.derivative(type_=("1"), var=x, phi=1.5)
        self.assertAlmostEqual(d, 3.0)
        d = q.derivative(type_=("1"), var=y, phi=3.0)
        self.assertAlmostEqual(d, 0.375)

        # test invalid derivative w.r.t. scalar
        a = 2.5
        q.coeff["1"]["m"] = a
        with self.assertRaises(TypeError):
            d = q.derivative(type_=("1"), var=a, phi=2.0)

        # test with respect to independent variable which is
        # related to a SameAs variable
        r = TwoVarPot(types=("1",), params=("x", "y"))

        r.coeff["1"]["x"] = x
        r.coeff["1"]["y"] = relentless.model.variable.SameAs(x)
        d = r.derivative(type_=("1"), var=x, phi=3.0)
        self.assertAlmostEqual(d, 15.0)

        r.coeff["1"]["y"] = x
        r.coeff["1"]["x"] = relentless.model.variable.SameAs(x)
        d = r.derivative(type_=("1"), var=x, phi=3.0)
        self.assertAlmostEqual(d, 15.0)

    def test_iteration(self):
        """Test iteration on typesPotential object"""
        p = LinPotDihedral(types=("1",), params=("m",))
        for types in p.coeff:
            p.coeff[types]["m"] = 2.0

        self.assertEqual(dict(p.coeff["1"]), {"m": 2.0})

    def test_json(self):
        """Test saving to file"""
        p = LinPotDihedral(types=("1",), params=("m"))
        p.coeff["1"]["m"] = 2.0

        data = p.to_json()
        self.assertEqual(data["id"], p.id)
        self.assertEqual(data["name"], p.name)

        p2 = LinPotDihedral.from_json(data)
        self.assertEqual(p2.coeff["1"]["m"], 2.0)

    def test_save(self):
        """Test saving to file"""
        temp = tempfile.NamedTemporaryFile()
        p = LinPotDihedral(types=("1",), params=("m"))
        p.coeff["1"]["m"] = 2.0
        p.save(temp.name)

        p2 = LinPotDihedral.from_file(temp.name)
        self.assertEqual(p2.coeff["1"]["m"], 2.0)

        temp.close()


class test_OPLSDihedral(unittest.TestCase):
    """Unit tests for relentless.model.potential.OPLSDihedral"""

    def test_init(self):
        """Test creation from data"""
        opls_dihedral = relentless.model.potential.OPLSDihedral(types=("1",))
        coeff = relentless.model.potential.DihedralParameters(
            types=("1",),
            params=(
                "k1",
                "k2",
                "k3",
                "k4",
            ),
        )
        self.assertCountEqual(opls_dihedral.coeff.types, coeff.types)
        self.assertCountEqual(opls_dihedral.coeff.params, coeff.params)

    def test_energy(self):
        """Test _energy method"""
        opls_dihedral = relentless.model.potential.OPLSDihedral(types=("1",))
        # general CT-CT-CT-CT from Watkins Phys Chem A, 105, 4118-4125 (2001).
        opls_dihedral.coeff["1"].update(k1=6.622, k2=0.948, k3=-1.388, k4=-2.118)
        # test scalar phi
        phi_input = numpy.pi / 4
        u_actual = 3.80496265865
        u = opls_dihedral.energy(type_=("1"), phi=phi_input)
        self.assertAlmostEqual(u, u_actual)

        # test array phi
        phi_input = numpy.array([0.0, numpy.pi / 2, numpy.pi])
        u_actual = numpy.array([5.234, 3.565, 0.0])
        u = opls_dihedral.energy(type_=("1"), phi=phi_input)
        numpy.testing.assert_allclose(u, u_actual)

        # test with invalid phi less than -pi
        with self.assertRaises(ValueError):
            opls_dihedral.energy(type_=("1"), phi=-3.5)

        # test with invalid phi greater than pi
        with self.assertRaises(ValueError):
            opls_dihedral.energy(type_=("1"), phi=3.5)

    def test_force(self):
        """Test _force method"""
        opls_dihedral = relentless.model.potential.OPLSDihedral(types=("1",))
        # general CT-CT-CT-CT from Watkins Phys Chem A, 105, 4118-4125 (2001).
        opls_dihedral.coeff["1"].update(k1=6.622, k2=0.948, k3=-1.388, k4=-2.118)

        # test scalar r
        phi_input = numpy.pi / 4
        f_actual = -0.0789657659217
        f = opls_dihedral.force(type_=("1"), phi=phi_input)
        self.assertAlmostEqual(f, f_actual)

        # test array r
        phi_input = numpy.array([0.0, numpy.pi / 2, numpy.pi])
        f_actual = numpy.array([0.0, 5.393, 0.0])
        f = opls_dihedral.force(type_=("1"), phi=phi_input)
        numpy.testing.assert_allclose(f, f_actual, atol=1e-14)

        # test with invalid phi less than -pi
        with self.assertRaises(ValueError):
            opls_dihedral.force(type_=("1"), phi=-3.5)

        # test with invalid phi greater than pi
        with self.assertRaises(ValueError):
            opls_dihedral.force(type_=("1"), phi=3.5)

    def test_derivative(self):
        """Test _derivative method"""
        opls_dihedral = relentless.model.potential.OPLSDihedral(types=("1",))

        # w.r.t. k1
        # test scalar r
        phi_input = numpy.pi / 4
        d_actual = 0.853553390593
        d = opls_dihedral._derivative(
            param="k1", phi=phi_input, k1=6.622, k2=0.948, k3=-1.388, k4=-2.118
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([1.0, 0.5, 0.0])
        d = opls_dihedral._derivative(
            param="k1", phi=phi_input, k1=6.622, k2=0.948, k3=-1.388, k4=-2.118
        )
        numpy.testing.assert_allclose(d, d_actual)

        # w.r.t. k2
        # test scalar r
        phi_input = numpy.pi / 4
        d_actual = -0.5
        d = opls_dihedral._derivative(
            param="k2", phi=phi_input, k1=6.622, k2=0.948, k3=-1.388, k4=-2.118
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([-1.0, 0.0, -1.0])
        d = opls_dihedral._derivative(
            param="k2", phi=phi_input, k1=6.622, k2=0.948, k3=-1.388, k4=-2.118
        )
        numpy.testing.assert_allclose(d, d_actual)

        # w.r.t. k3
        # test scalar r
        phi_input = numpy.pi / 4
        d_actual = 0.146446609407
        d = opls_dihedral._derivative(
            param="k3", phi=phi_input, k1=6.622, k2=0.948, k3=-1.388, k4=-2.118
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([1.0, 0.5, 0.0])
        d = opls_dihedral._derivative(
            param="k3", phi=phi_input, k1=6.622, k2=0.948, k3=-1.388, k4=-2.118
        )
        numpy.testing.assert_allclose(d, d_actual)

        # w.r.t. k4
        # test scalar r
        phi_input = numpy.pi / 4
        d_actual = 0.0
        d = opls_dihedral._derivative(
            param="k4", phi=phi_input, k1=6.622, k2=0.948, k3=-1.388, k4=-2.118
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([-1.0, -1.0, -1])
        d = opls_dihedral._derivative(
            param="k4", phi=phi_input, k1=6.622, k2=0.948, k3=-1.388, k4=-2.118
        )
        numpy.testing.assert_allclose(d, d_actual)

        # test invalid param
        with self.assertRaises(ValueError):
            opls_dihedral._derivative(
                param="k0", phi=phi_input, k1=6.622, k2=0.948, k3=-1.388, k4=-2.118
            )

    def test_json(self):
        opls_dihedral = relentless.model.potential.OPLSDihedral(types=("1",))
        opls_dihedral.coeff["1"].update(k1=6.622, k2=0.948, k3=-1.388, k4=-2.118)
        data = opls_dihedral.to_json()

        opls_dihedral2 = relentless.model.potential.OPLSDihedral.from_json(data)
        self.assertEqual(opls_dihedral2.coeff["1"]["k1"], 6.622)
        self.assertEqual(opls_dihedral2.coeff["1"]["k2"], 0.948)
        self.assertEqual(opls_dihedral2.coeff["1"]["k3"], -1.388)
        self.assertEqual(opls_dihedral2.coeff["1"]["k4"], -2.118)


class test_RyckaertBellemansDihedral(unittest.TestCase):
    """Unit tests for relentless.model.potential.RyckaertBellemansDihedral"""

    def test_init(self):
        """Test creation from data"""
        ryckaert_bellemans_dihedral = (
            relentless.model.potential.RyckaertBellemansDihedral(types=("1",))
        )
        coeff = relentless.model.potential.DihedralParameters(
            types=("1",),
            params=(
                "c0",
                "c1",
                "c2",
                "c3",
                "c4",
                "c5",
            ),
        )
        self.assertCountEqual(ryckaert_bellemans_dihedral.coeff.types, coeff.types)
        self.assertCountEqual(ryckaert_bellemans_dihedral.coeff.params, coeff.params)

    def test_energy(self):
        """Test _energy method"""
        ryckaert_bellemans_dihedral = (
            relentless.model.potential.RyckaertBellemansDihedral(types=("1",))
        )
        # example of constants from GROMACS 2024.3 documentation
        ryckaert_bellemans_dihedral.coeff["1"].update(
            c0=9.28, c1=12.16, c2=-13.12, c3=-3.06, c4=26.24, c5=-31.5
        )
        # test scalar r
        phi_input = numpy.pi / 4
        u_actual = 7.33192081783
        u = ryckaert_bellemans_dihedral.energy(type_=("1"), phi=phi_input)
        self.assertAlmostEqual(u, u_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        u_actual = numpy.array([44.8, 9.28, 0])
        u = ryckaert_bellemans_dihedral.energy(type_=("1"), phi=phi_input)
        numpy.testing.assert_allclose(u, u_actual, atol=1e-14)

        # test with invalid phi less than -pi
        with self.assertRaises(ValueError):
            ryckaert_bellemans_dihedral.energy(type_=("1"), phi=-3.5)

        # test with invalid phi greater than pi
        with self.assertRaises(ValueError):
            ryckaert_bellemans_dihedral.energy(type_=("1"), phi=3.5)

    def test_force(self):
        """Test _force method"""
        ryckaert_bellemans_dihedral = (
            relentless.model.potential.RyckaertBellemansDihedral(types=("1",))
        )
        # example of constants from GROMACS 2024.3 documentation
        ryckaert_bellemans_dihedral.coeff["1"].update(
            c0=9.28, c1=12.16, c2=-13.12, c3=-3.06, c4=26.24, c5=-31.5
        )
        # test scalar r
        phi_input = numpy.pi / 4
        f_actual = 7.76720166642
        f = ryckaert_bellemans_dihedral.force(type_=("1"), phi=phi_input)
        self.assertAlmostEqual(f, f_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        f_actual = numpy.array([0.0, -12.16, 0.0])
        f = ryckaert_bellemans_dihedral.force(type_=("1"), phi=phi_input)
        numpy.testing.assert_allclose(f, f_actual, atol=1e-14)

        # test with invalid phi less than -pi
        with self.assertRaises(ValueError):
            ryckaert_bellemans_dihedral.force(type_=("1"), phi=-3.5)

        # test with invalid phi greater than pi
        with self.assertRaises(ValueError):
            ryckaert_bellemans_dihedral.force(type_=("1"), phi=3.5)

    def test_derivative(self):
        """Test _derivative method"""
        cosinesquared_dihedral = relentless.model.potential.RyckaertBellemansDihedral(
            types=("1",)
        )
        # w.r.t. c0
        # test scalar r
        phi_input = numpy.pi / 4
        d_actual = 1
        d = cosinesquared_dihedral._derivative(
            param="c0",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([1.0, 1.0, 1.0])
        d = cosinesquared_dihedral._derivative(
            param="c0",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        numpy.testing.assert_allclose(d, d_actual, atol=1e-14)

        # w.r.t. c1
        # test scalar r
        phi_input = numpy.pi / 4
        d_actual = -0.707106781187
        d = cosinesquared_dihedral._derivative(
            param="c1",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([-1.0, 0.0, 1.0])
        d = cosinesquared_dihedral._derivative(
            param="c1",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        numpy.testing.assert_allclose(d, d_actual, atol=1e-14)

        # w.r.t. c2
        # test scalar r
        phi_input = numpy.pi / 4
        d_actual = 0.5
        d = cosinesquared_dihedral._derivative(
            param="c2",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([1.0, 0.0, 1.0])
        d = cosinesquared_dihedral._derivative(
            param="c2",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        numpy.testing.assert_allclose(d, d_actual, atol=1e-14)

        # w.r.t. c3
        # test scalar r
        phi_input = numpy.pi / 4
        d_actual = -0.353553390593
        d = cosinesquared_dihedral._derivative(
            param="c3",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([-1.0, 0.0, 1.0])
        d = cosinesquared_dihedral._derivative(
            param="c3",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        numpy.testing.assert_allclose(d, d_actual, atol=1e-14)

        # w.r.t. c4
        # test scalar r
        phi_input = numpy.pi / 4
        d_actual = 0.25
        d = cosinesquared_dihedral._derivative(
            param="c4",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([1.0, 0.0, 1.0])
        d = cosinesquared_dihedral._derivative(
            param="c4",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        numpy.testing.assert_allclose(d, d_actual, atol=1e-14)

        # w.r.t. c5
        # test scalar r
        phi_input = numpy.pi / 4
        d_actual = -0.176776695297
        d = cosinesquared_dihedral._derivative(
            param="c5",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        self.assertAlmostEqual(d, d_actual)

        # test array r
        phi_input = numpy.array([0, numpy.pi / 2, numpy.pi])
        d_actual = numpy.array([-1.0, 0.0, 1.0])
        d = cosinesquared_dihedral._derivative(
            param="c5",
            phi=phi_input,
            c0=9.28,
            c1=12.16,
            c2=-13.12,
            c3=-3.06,
            c4=26.24,
            c5=-31.5,
        )
        numpy.testing.assert_allclose(d, d_actual, atol=1e-14)

        # test invalid param
        with self.assertRaises(ValueError):
            cosinesquared_dihedral._derivative(
                param="co",
                phi=phi_input,
                c0=9.28,
                c1=12.16,
                c2=-13.12,
                c3=-3.06,
                c4=26.24,
                c5=-31.5,
            )

    def test_json(self):
        ryckaert_bellemans_dihedral = (
            relentless.model.potential.RyckaertBellemansDihedral(types=("1",))
        )
        ryckaert_bellemans_dihedral.coeff["1"].update(
            c0=9.28, c1=12.16, c2=-13.12, c3=-3.06, c4=26.24, c5=-31.5
        )
        data = ryckaert_bellemans_dihedral.to_json()

        ryckaert_bellemans_dihedral2 = (
            relentless.model.potential.RyckaertBellemansDihedral.from_json(data)
        )
        self.assertEqual(ryckaert_bellemans_dihedral2.coeff["1"]["c0"], 9.28)
        self.assertEqual(ryckaert_bellemans_dihedral2.coeff["1"]["c1"], 12.16)
        self.assertEqual(ryckaert_bellemans_dihedral2.coeff["1"]["c2"], -13.12)
        self.assertEqual(ryckaert_bellemans_dihedral2.coeff["1"]["c3"], -3.06)
        self.assertEqual(ryckaert_bellemans_dihedral2.coeff["1"]["c4"], 26.24)
        self.assertEqual(ryckaert_bellemans_dihedral2.coeff["1"]["c5"], -31.5)


class test_DihedralSpline(unittest.TestCase):
    """Unit tests for relentless.model.potential.DihedralSpline"""

    def test_init(self):
        """Test creation from data"""
        # test diff mode
        s = relentless.model.potential.DihedralSpline(types=("1",), num_knots=3)
        self.assertEqual(s.num_knots, 3)
        self.assertEqual(s.mode, "diff")
        coeff = relentless.model.potential.DihedralParameters(
            types=("1",),
            params=(
                "dphi-0",
                "dphi-1",
                "dphi-2",
                "diff-0",
                "diff-1",
                "diff-2",
            ),
        )
        self.assertCountEqual(s.coeff.types, coeff.types)
        self.assertCountEqual(s.coeff.params, coeff.params)

        # test value mode
        s = relentless.model.potential.DihedralSpline(
            types=("1",), num_knots=3, mode="value"
        )
        self.assertEqual(s.num_knots, 3)
        self.assertEqual(s.mode, "value")
        coeff = relentless.model.potential.DihedralParameters(
            types=("1",),
            params=(
                "dphi-0",
                "dphi-1",
                "dphi-2",
                "value-0",
                "value-1",
                "value-2",
            ),
        )
        self.assertCountEqual(s.coeff.types, coeff.types)
        self.assertCountEqual(s.coeff.params, coeff.params)

        # test invalid number of knots
        with self.assertRaises(ValueError):
            s = relentless.model.potential.DihedralSpline(types=("1",), num_knots=1)

        # test invalid mode
        with self.assertRaises(ValueError):
            s = relentless.model.potential.DihedralSpline(
                types=("1",), num_knots=3, mode="val"
            )

    def test_from_array(self):
        """Test from_array method and knots generator"""
        phi_arr = [1, 2, 3]
        u_arr = [1, 2, 3]

        # test that bounds are enforced
        s = relentless.model.potential.DihedralSpline(types=("1",), num_knots=3)
        with self.assertRaises(ValueError):
            s.from_array(types=("1"), phi=phi_arr, u=u_arr)


if __name__ == "__main__":
    unittest.main()
