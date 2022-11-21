"""Unit tests for core.extent module."""
import unittest

import numpy

import relentless

class test_TriclinicBox(unittest.TestCase):
    """Unit tests for relentless.model.TriclinicBox"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction, LAMMPS convention
        t = relentless.model.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,
                                           convention='LAMMPS')
        numpy.testing.assert_allclose(t.a, numpy.array([1,0,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2,0]))
        numpy.testing.assert_allclose(t.c, numpy.array([0.75,2.25,3]))
        self.assertAlmostEqual(t.extent, 6)

        # test valid construction, HOOMD convention
        t = relentless.model.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=0.5,xz=0.25,yz=0.75,
                                           convention='HOOMD')
        numpy.testing.assert_allclose(t.a, numpy.array([1,0,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2,0]))
        numpy.testing.assert_allclose(t.c, numpy.array([0.75,2.25,3]))
        self.assertAlmostEqual(t.extent, 6)

        # test invalid constructions
        with self.assertRaises(ValueError):
            t = relentless.model.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,convention='not-real')
        with self.assertRaises(ValueError):
            t = relentless.model.TriclinicBox(Lx=-1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,
                                               convention='LAMMPS')

    def test_coordinate_transform(self):
        t = relentless.model.TriclinicBox(Lx=1, Ly=2, Lz=3, xy=1, xz=0.75, yz=2.25)

        # check upper/lower bounds
        numpy.testing.assert_allclose(t.coordinate_to_fraction(t.low), [0,0,0])
        numpy.testing.assert_allclose(t.coordinate_to_fraction(t.high), [1,1,1])

        # origin should be at center of box
        x = t.coordinate_to_fraction([0,0,0])
        numpy.testing.assert_allclose(x, [0.5,0.5,0.5])
        r = t.fraction_to_coordinate(x)
        numpy.testing.assert_allclose(r, [0,0,0])

        # do a few made up coordinates, and ensure there & back works
        r = [[-1,0.5,-0.25],[3,-2,1], [100,-200,300]]
        r_2 = t.fraction_to_coordinate(t.coordinate_to_fraction(r))
        numpy.testing.assert_allclose(r_2, r)

        # go the other way
        x = [[0.2,0.3,0.4],[0,1,0.9],[-1,100,-3]]
        x_2 = t.fraction_to_coordinate(t.coordinate_to_fraction(x))
        numpy.testing.assert_allclose(x_2, x)

        # make some fractions and wrap them back in
        x = [[0.1,0.2,0.3],[1.1,-0.1,1.2],[3.1,-1.1,-2.2]]
        r = t.fraction_to_coordinate(x)
        r = t.wrap(r)
        x = t.coordinate_to_fraction(r)
        numpy.testing.assert_allclose(x, [[0.1,0.2,0.3],[0.1,0.9,0.2],[0.1,0.9,0.8]])

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        # test LAMMPS convention
        c = relentless.model.TriclinicBox(Lx=3,Ly=4,Lz=5,xy=2,xz=3,yz=4,
                                           convention='LAMMPS')
        data = c.to_json()
        c_ = relentless.model.TriclinicBox.from_json(data)
        self.assertIsInstance(c_, relentless.model.TriclinicBox)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.extent, c_.extent)

        # test HOOMD convention
        c = relentless.model.TriclinicBox(Lx=3,Ly=4,Lz=5,xy=2,xz=3,yz=4,
                                           convention='HOOMD')
        data = c.to_json()
        c_ = relentless.model.TriclinicBox.from_json(data)
        self.assertIsInstance(c_, relentless.model.TriclinicBox)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.extent, c_.extent)

class test_Cuboid(unittest.TestCase):
    """Unit tests for relentless.model.Cuboid"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.model.Cuboid(Lx=3,Ly=4,Lz=5)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,4,0]))
        numpy.testing.assert_allclose(c.c, numpy.array([0,0,5]))
        self.assertAlmostEqual(c.extent, 60)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.model.Cuboid(Lx=-3,Ly=4,Lz=5)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.model.Cuboid(Lx=3,Ly=4,Lz=5)
        data = c.to_json()
        c_ = relentless.model.Cuboid.from_json(data)
        self.assertIsInstance(c_, relentless.model.Cuboid)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.extent, c_.extent)

class test_Cube(unittest.TestCase):
    """Unit tests for relentless.model.Cube"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.model.Cube(L=3)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,3,0]))
        numpy.testing.assert_allclose(c.c, numpy.array([0,0,3]))
        self.assertAlmostEqual(c.extent, 27)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.model.Cube(L=-1)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.model.Cube(L=3)
        data = c.to_json()
        c_ = relentless.model.Cube.from_json(data)
        self.assertIsInstance(c_, relentless.model.Cube)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.extent, c_.extent)

class test_ObliqueArea(unittest.TestCase):
    """Unit tests for relentless.model.ObliqueArea"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction, LAMMPS convention
        t = relentless.model.ObliqueArea(Lx=1,Ly=2,xy=1, convention='LAMMPS')
        numpy.testing.assert_allclose(t.a, numpy.array([1,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2]))
        self.assertAlmostEqual(t.extent, 2)

        # test valid construction, HOOMD convention
        t = relentless.model.ObliqueArea(Lx=1,Ly=2,xy=0.5,
                                           convention='HOOMD')
        numpy.testing.assert_allclose(t.a, numpy.array([1,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2]))
        self.assertAlmostEqual(t.extent, 2)    

        # test invalid constructions
        with self.assertRaises(ValueError):
            t = relentless.model.ObliqueArea(Lx=1,Ly=2,xy=1,convention='not-real')
        with self.assertRaises(ValueError):
            t = relentless.model.ObliqueArea(Lx=-1,Ly=2,xy=1,
                                               convention='LAMMPS')

    def test_coordinate_transform(self):
        t = relentless.model.ObliqueArea(Lx=1, Ly=2, xy=1)

        # check upper/lower bounds
        numpy.testing.assert_allclose(t.coordinate_to_fraction(t.low), [0,0])
        numpy.testing.assert_allclose(t.coordinate_to_fraction(t.high), [1,1])

        # origin should be at center of box
        x = t.coordinate_to_fraction([0,0])
        numpy.testing.assert_allclose(x, [0.5,0.5])
        r = t.fraction_to_coordinate(x)
        numpy.testing.assert_allclose(r, [0,0])

        # do a few made up coordinates, and ensure there & back works
        r = [[-1,0.5],[3,-2], [100,-200]]
        r_2 = t.fraction_to_coordinate(t.coordinate_to_fraction(r))
        numpy.testing.assert_allclose(r_2, r)

        # go the other way
        x = [[0.2,0.3],[0,1],[-1,100]]
        x_2 = t.fraction_to_coordinate(t.coordinate_to_fraction(x))
        numpy.testing.assert_allclose(x_2, x)

        # make some fractions and wrap them back in
        x = [[0.1,0.2],[1.1,-0.1],[3.1,-2.2]]
        r = t.fraction_to_coordinate(x)
        r = t.wrap(r)
        x = t.coordinate_to_fraction(r)
        numpy.testing.assert_allclose(x, [[0.1,0.2],[0.1,0.9],[0.1,0.8]])

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        # test LAMMPS convention
        c = relentless.model.ObliqueArea(Lx=3,Ly=4,xy=2,
                                           convention='LAMMPS')
        data = c.to_json()
        c_ = relentless.model.ObliqueArea.from_json(data)
        self.assertIsInstance(c_, relentless.model.ObliqueArea)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.extent, c_.extent)

        # test HOOMD convention
        c = relentless.model.ObliqueArea(Lx=3,Ly=4,xy=2,
                                           convention='HOOMD')
        data = c.to_json()
        c_ = relentless.model.ObliqueArea.from_json(data)
        self.assertIsInstance(c_, relentless.model.ObliqueArea)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.extent, c_.extent) 

class test_Rectangle(unittest.TestCase):
    """Unit tests for relentless.model.Rectangle"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.model.Rectangle(Lx=3,Ly=4)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,4]))
        self.assertAlmostEqual(c.extent, 12)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.model.Rectangle(Lx=-3,Ly=4)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.model.Rectangle(Lx=3,Ly=4)
        data = c.to_json()
        c_ = relentless.model.Rectangle.from_json(data)
        self.assertIsInstance(c_, relentless.model.Rectangle)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.extent, c_.extent)

class test_Square(unittest.TestCase):
    """Unit tests for relentless.model.Square"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.model.Square(L=3)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,3]))
        self.assertAlmostEqual(c.extent, 9)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.model.Square(L=-1)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.model.Square(L=3)
        data = c.to_json()
        c_ = relentless.model.Square.from_json(data)
        self.assertIsInstance(c_, relentless.model.Square)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.extent, c_.extent)

if __name__ == '__main__':
    unittest.main()
