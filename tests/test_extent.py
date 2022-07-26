"""Unit tests for core.extent module."""
import unittest

import numpy

import relentless

class test_Parallelepiped(unittest.TestCase):
    """Unit tests for relentless.extent.Parallelepiped"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        p = relentless.extent.Parallelepiped(a=(1,2,1),b=(3,4,5),c=(9,9,0))
        numpy.testing.assert_allclose(p.a, numpy.array([1,2,1]))
        numpy.testing.assert_allclose(p.b, numpy.array([3,4,5]))
        numpy.testing.assert_allclose(p.c, numpy.array([9,9,0]))
        self.assertAlmostEqual(p.volume, 36)

        # test invalid construction
        with self.assertRaises(TypeError):
            p = relentless.extent.Parallelepiped(a=(1,2,1),b=(3,4,5),c=(9,9))
        with self.assertRaises(ValueError):
            p = relentless.extent.Parallelepiped(a=(-1,-2,-1),b=(3,-4,5),c=(2,4,1))

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        p = relentless.extent.Parallelepiped(a=(1,2,1),b=(3,4,5),c=(9,9,0))
        data = p.to_json()
        p_ = relentless.extent.Parallelepiped.from_json(data)
        self.assertIsInstance(p_, relentless.extent.Parallelepiped)
        numpy.testing.assert_allclose(p.a, p_.a)
        numpy.testing.assert_allclose(p.b, p_.b)
        numpy.testing.assert_allclose(p.c, p_.c)
        self.assertAlmostEqual(p.volume, p_.volume)

class test_TriclinicBox(unittest.TestCase):
    """Unit tests for relentless.extent.TriclinicBox"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction, LAMMPS convention
        t = relentless.extent.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,
                                           convention=relentless.extent.TriclinicBox.Convention.LAMMPS)
        numpy.testing.assert_allclose(t.a, numpy.array([1,0,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2,0]))
        numpy.testing.assert_allclose(t.c, numpy.array([0.75,2.25,3]))
        self.assertAlmostEqual(t.volume, 6)

        # test valid construction, HOOMD convention
        t = relentless.extent.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=0.5,xz=0.25,yz=0.75,
                                           convention=relentless.extent.TriclinicBox.Convention.HOOMD)
        numpy.testing.assert_allclose(t.a, numpy.array([1,0,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2,0]))
        numpy.testing.assert_allclose(t.c, numpy.array([0.75,2.25,3]))
        self.assertAlmostEqual(t.volume, 6)

        # test invalid constructions
        with self.assertRaises(ValueError):
            t = relentless.extent.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,convention='LAMMPS')
        with self.assertRaises(ValueError):
            t = relentless.extent.TriclinicBox(Lx=-1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,
                                               convention=relentless.extent.TriclinicBox.Convention.LAMMPS)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        # test LAMMPS convention
        c = relentless.extent.TriclinicBox(Lx=3,Ly=4,Lz=5,xy=2,xz=3,yz=4,
                                           convention=relentless.extent.TriclinicBox.Convention.LAMMPS)
        data = c.to_json()
        c_ = relentless.extent.TriclinicBox.from_json(data)
        self.assertIsInstance(c_, relentless.extent.TriclinicBox)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

        # test HOOMD convention
        c = relentless.extent.TriclinicBox(Lx=3,Ly=4,Lz=5,xy=2,xz=3,yz=4,
                                           convention=relentless.extent.TriclinicBox.Convention.HOOMD)
        data = c.to_json()
        c_ = relentless.extent.TriclinicBox.from_json(data)
        self.assertIsInstance(c_, relentless.extent.TriclinicBox)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

class test_Cuboid(unittest.TestCase):
    """Unit tests for relentless.extent.Cuboid"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.extent.Cuboid(Lx=3,Ly=4,Lz=5)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,4,0]))
        numpy.testing.assert_allclose(c.c, numpy.array([0,0,5]))
        self.assertAlmostEqual(c.volume, 60)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.extent.Cuboid(Lx=-3,Ly=4,Lz=5)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.extent.Cuboid(Lx=3,Ly=4,Lz=5)
        data = c.to_json()
        c_ = relentless.extent.Cuboid.from_json(data)
        self.assertIsInstance(c_, relentless.extent.Cuboid)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

class test_Cube(unittest.TestCase):
    """Unit tests for relentless.extent.Cube"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.extent.Cube(L=3)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,3,0]))
        numpy.testing.assert_allclose(c.c, numpy.array([0,0,3]))
        self.assertAlmostEqual(c.volume, 27)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.extent.Cube(L=-1)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.extent.Cube(L=3)
        data = c.to_json()
        c_ = relentless.extent.Cube.from_json(data)
        self.assertIsInstance(c_, relentless.extent.Cube)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

class test_Parallelogram(unittest.TestCase):
    """Unit tests for relentless.extent.Parallelogram"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        p = relentless.extent.Parallelogram(a=(2,5),b=(1,4))
        numpy.testing.assert_allclose(p.a, numpy.array([2,5]))
        numpy.testing.assert_allclose(p.b, numpy.array([1,4]))
        self.assertAlmostEqual(p.area, 3)

        # test invalid construction
        with self.assertRaises(TypeError):
            p = relentless.extent.Parallelogram(a=(1,2,1),b=(3,4))
        with self.assertRaises(ValueError):
            p = relentless.extent.Parallelogram(a=(1,2),b=(2,4))

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        p = relentless.extent.Parallelogram(a=(1,2),b=(3,4))
        data = p.to_json()
        p_ = relentless.extent.Parallelogram.from_json(data)
        self.assertIsInstance(p_, relentless.extent.Parallelogram)
        numpy.testing.assert_allclose(p.a, p_.a)
        numpy.testing.assert_allclose(p.b, p_.b)
        self.assertAlmostEqual(p.area, p_.area)

class test_ObliqueArea(unittest.TestCase):
    """Unit tests for relentless.extent.ObliqueArea"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction, LAMMPS convention
        t = relentless.extent.ObliqueArea(Lx=1,Ly=2,xy=1, convention=relentless.extent.ObliqueArea.Convention.LAMMPS)
        numpy.testing.assert_allclose(t.a, numpy.array([1,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2]))
        self.assertAlmostEqual(t.area, 2)

        # test valid construction, HOOMD convention
        t = relentless.extent.ObliqueArea(Lx=1,Ly=2,xy=0.5,
                                           convention=relentless.extent.ObliqueArea.Convention.HOOMD)
        numpy.testing.assert_allclose(t.a, numpy.array([1,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2]))
        self.assertAlmostEqual(t.area, 2)    

        # test invalid constructions
        with self.assertRaises(ValueError):
            t = relentless.extent.ObliqueArea(Lx=1,Ly=2,xy=1,convention='LAMMPS')
        with self.assertRaises(ValueError):
            t = relentless.extent.ObliqueArea(Lx=-1,Ly=2,xy=1,
                                               convention=relentless.extent.TriclinicBox.Convention.LAMMPS)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        # test LAMMPS convention
        c = relentless.extent.ObliqueArea(Lx=3,Ly=4,xy=2,
                                           convention=relentless.extent.ObliqueArea.Convention.LAMMPS)
        data = c.to_json()
        c_ = relentless.extent.ObliqueArea.from_json(data)
        self.assertIsInstance(c_, relentless.extent.ObliqueArea)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.area, c_.area)

        # test HOOMD convention
        c = relentless.extent.ObliqueArea(Lx=3,Ly=4,xy=2,
                                           convention=relentless.extent.ObliqueArea.Convention.HOOMD)
        data = c.to_json()
        c_ = relentless.extent.ObliqueArea.from_json(data)
        self.assertIsInstance(c_, relentless.extent.ObliqueArea)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.area, c_.area) 

class test_Rectangle(unittest.TestCase):
    """Unit tests for relentless.extent.Rectangle"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.extent.Rectangle(Lx=3,Ly=4)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,4]))
        self.assertAlmostEqual(c.area, 12)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.extent.Rectangle(Lx=-3,Ly=4)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.extent.Rectangle(Lx=3,Ly=4)
        data = c.to_json()
        c_ = relentless.extent.Rectangle.from_json(data)
        self.assertIsInstance(c_, relentless.extent.Rectangle)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.area, c_.area)

class test_Square(unittest.TestCase):
    """Unit tests for relentless.extent.Square"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.extent.Square(L=3)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,3]))
        self.assertAlmostEqual(c.area, 9)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.extent.Square(L=-1)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.extent.Square(L=3)
        data = c.to_json()
        c_ = relentless.extent.Square.from_json(data)
        self.assertIsInstance(c_, relentless.extent.Square)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.area, c_.area)

if __name__ == '__main__':
    unittest.main()
