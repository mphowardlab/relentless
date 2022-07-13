"""Unit tests for core.volume module."""
import unittest

import numpy

import relentless

class test_Parallelepiped(unittest.TestCase):
    """Unit tests for relentless.volume.Parallelepiped"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        p = relentless.volume.Parallelepiped(a=(1,2,1),b=(3,4,5),c=(9,9,0))
        numpy.testing.assert_allclose(p.a, numpy.array([1,2,1]))
        numpy.testing.assert_allclose(p.b, numpy.array([3,4,5]))
        numpy.testing.assert_allclose(p.c, numpy.array([9,9,0]))
        self.assertAlmostEqual(p.volume, 36)

        # test invalid construction
        with self.assertRaises(TypeError):
            p = relentless.volume.Parallelepiped(a=(1,2,1),b=(3,4,5),c=(9,9))
        with self.assertRaises(ValueError):
            p = relentless.volume.Parallelepiped(a=(-1,-2,-1),b=(3,-4,5),c=(2,4,1))

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        p = relentless.volume.Parallelepiped(a=(1,2,1),b=(3,4,5),c=(9,9,0))
        data = p.to_json()
        p_ = relentless.volume.Parallelepiped.from_json(data)
        self.assertIsInstance(p_, relentless.volume.Parallelepiped)
        numpy.testing.assert_allclose(p.a, p_.a)
        numpy.testing.assert_allclose(p.b, p_.b)
        numpy.testing.assert_allclose(p.c, p_.c)
        self.assertAlmostEqual(p.volume, p_.volume)

class test_TriclinicBox(unittest.TestCase):
    """Unit tests for relentless.volume.TriclinicBox"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction, LAMMPS convention
        t = relentless.volume.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,
                                           convention=relentless.volume.TriclinicBox.Convention.LAMMPS)
        numpy.testing.assert_allclose(t.a, numpy.array([1,0,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2,0]))
        numpy.testing.assert_allclose(t.c, numpy.array([0.75,2.25,3]))
        self.assertAlmostEqual(t.volume, 6)

        # test valid construction, HOOMD convention
        t = relentless.volume.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=0.5,xz=0.25,yz=0.75,
                                           convention=relentless.volume.TriclinicBox.Convention.HOOMD)
        numpy.testing.assert_allclose(t.a, numpy.array([1,0,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2,0]))
        numpy.testing.assert_allclose(t.c, numpy.array([0.75,2.25,3]))
        self.assertAlmostEqual(t.volume, 6)

        # test invalid constructions
        with self.assertRaises(ValueError):
            t = relentless.volume.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,convention='LAMMPS')
        with self.assertRaises(ValueError):
            t = relentless.volume.TriclinicBox(Lx=-1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,
                                               convention=relentless.volume.TriclinicBox.Convention.LAMMPS)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        # test LAMMPS convention
        c = relentless.volume.TriclinicBox(Lx=3,Ly=4,Lz=5,xy=2,xz=3,yz=4,
                                           convention=relentless.volume.TriclinicBox.Convention.LAMMPS)
        data = c.to_json()
        c_ = relentless.volume.TriclinicBox.from_json(data)
        self.assertIsInstance(c_, relentless.volume.TriclinicBox)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

        # test HOOMD convention
        c = relentless.volume.TriclinicBox(Lx=3,Ly=4,Lz=5,xy=2,xz=3,yz=4,
                                           convention=relentless.volume.TriclinicBox.Convention.HOOMD)
        data = c.to_json()
        c_ = relentless.volume.TriclinicBox.from_json(data)
        self.assertIsInstance(c_, relentless.volume.TriclinicBox)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

class test_Cuboid(unittest.TestCase):
    """Unit tests for relentless.volume.Cuboid"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.volume.Cuboid(Lx=3,Ly=4,Lz=5)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,4,0]))
        numpy.testing.assert_allclose(c.c, numpy.array([0,0,5]))
        self.assertAlmostEqual(c.volume, 60)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.volume.Cuboid(Lx=-3,Ly=4,Lz=5)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.volume.Cuboid(Lx=3,Ly=4,Lz=5)
        data = c.to_json()
        c_ = relentless.volume.Cuboid.from_json(data)
        self.assertIsInstance(c_, relentless.volume.Cuboid)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

class test_Cube(unittest.TestCase):
    """Unit tests for relentless.volume.Cube"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.volume.Cube(L=3)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,3,0]))
        numpy.testing.assert_allclose(c.c, numpy.array([0,0,3]))
        self.assertAlmostEqual(c.volume, 27)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.volume.Cube(L=-1)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.volume.Cube(L=3)
        data = c.to_json()
        c_ = relentless.volume.Cube.from_json(data)
        self.assertIsInstance(c_, relentless.volume.Cube)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        numpy.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

class test_Parallelogram(unittest.TestCase):
    """Unit tests for relentless.volume.Parallelogram"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        p = relentless.volume.Parallelogram(a=(2,5),b=(1,4))
        numpy.testing.assert_allclose(p.a, numpy.array([2,5]))
        numpy.testing.assert_allclose(p.b, numpy.array([1,4]))
        self.assertAlmostEqual(p.area, 3)

        # test invalid construction
        with self.assertRaises(TypeError):
            p = relentless.volume.Parallelogram(a=(1,2,1),b=(3,4))
        with self.assertRaises(ValueError):
            p = relentless.volume.Parallelogram(a=(1,2),b=(2,4))

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        p = relentless.volume.Parallelogram(a=(1,2),b=(3,4))
        data = p.to_json()
        p_ = relentless.volume.Parallelogram.from_json(data)
        self.assertIsInstance(p_, relentless.volume.Parallelogram)
        numpy.testing.assert_allclose(p.a, p_.a)
        numpy.testing.assert_allclose(p.b, p_.b)
        self.assertAlmostEqual(p.area, p_.area)

class test_ObliqueArea(unittest.TestCase):
    """Unit tests for relentless.volume.TriclinicBox"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction, LAMMPS convention
        t = relentless.volume.ObliqueArea(Lx=1,Ly=2,xy=1, convention=relentless.volume.ObliqueArea.Convention.LAMMPS)
        numpy.testing.assert_allclose(t.a, numpy.array([1,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2]))
        self.assertAlmostEqual(t.area, 2)

        # test valid construction, HOOMD convention
        t = relentless.volume.ObliqueArea(Lx=1,Ly=2,xy=0.5,
                                           convention=relentless.volume.ObliqueArea.Convention.HOOMD)
        numpy.testing.assert_allclose(t.a, numpy.array([1,0]))
        numpy.testing.assert_allclose(t.b, numpy.array([1,2]))
        self.assertAlmostEqual(t.area, 2)    

        # test invalid constructions
        with self.assertRaises(ValueError):
            t = relentless.volume.ObliqueArea(Lx=1,Ly=2,xy=1,convention='LAMMPS')
        with self.assertRaises(ValueError):
            t = relentless.volume.ObliqueArea(Lx=-1,Ly=2,xy=1,
                                               convention=relentless.volume.TriclinicBox.Convention.LAMMPS)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        # test LAMMPS convention
        c = relentless.volume.ObliqueArea(Lx=3,Ly=4,xy=2,
                                           convention=relentless.volume.ObliqueArea.Convention.LAMMPS)
        data = c.to_json()
        c_ = relentless.volume.ObliqueArea.from_json(data)
        self.assertIsInstance(c_, relentless.volume.ObliqueArea)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.area, c_.area)

        # test HOOMD convention
        c = relentless.volume.ObliqueArea(Lx=3,Ly=4,xy=2,
                                           convention=relentless.volume.ObliqueArea.Convention.HOOMD)
        data = c.to_json()
        c_ = relentless.volume.ObliqueArea.from_json(data)
        self.assertIsInstance(c_, relentless.volume.ObliqueArea)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.area, c_.area) 

class test_Rectangle(unittest.TestCase):
    """Unit tests for relentless.volume.Rectangle"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.volume.Rectangle(Lx=3,Ly=4)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,4]))
        self.assertAlmostEqual(c.area, 12)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.volume.Rectangle(Lx=-3,Ly=4)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.volume.Rectangle(Lx=3,Ly=4)
        data = c.to_json()
        c_ = relentless.volume.Rectangle.from_json(data)
        self.assertIsInstance(c_, relentless.volume.Rectangle)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.area, c_.area)

class test_Square(unittest.TestCase):
    """Unit tests for relentless.volume.Square"""

    def test_init(self):
        """Test creation from data."""
        # test valid construction
        c = relentless.volume.Square(L=3)
        numpy.testing.assert_allclose(c.a, numpy.array([3,0]))
        numpy.testing.assert_allclose(c.b, numpy.array([0,3]))
        self.assertAlmostEqual(c.area, 9)

        # test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.volume.Square(L=-1)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.volume.Square(L=3)
        data = c.to_json()
        c_ = relentless.volume.Square.from_json(data)
        self.assertIsInstance(c_, relentless.volume.Square)
        numpy.testing.assert_allclose(c.a, c_.a)
        numpy.testing.assert_allclose(c.b, c_.b)
        self.assertAlmostEqual(c.area, c_.area)

if __name__ == '__main__':
    unittest.main()
