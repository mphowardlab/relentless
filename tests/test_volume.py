"""Unit tests for core.volume module."""
import unittest

import numpy as np

import relentless

class test_Parallelepiped(unittest.TestCase):
    """Unit tests for relentless.Parallelepiped"""

    def test_init(self):
        """Test creation from data."""
        #test valid construction
        p = relentless.Parallelepiped(a=(1,2,1),b=(3,4,5),c=(9,9,0))
        np.testing.assert_allclose(p.a, np.array([1,2,1]))
        np.testing.assert_allclose(p.b, np.array([3,4,5]))
        np.testing.assert_allclose(p.c, np.array([9,9,0]))
        self.assertAlmostEqual(p.volume, 36)

        #test invalid construction
        with self.assertRaises(TypeError):
            p = relentless.Parallelepiped(a=(1,2,1),b=(3,4,5),c=(9,9))
        with self.assertRaises(ValueError):
            p = relentless.Parallelepiped(a=(-1,-2,-1),b=(3,-4,5),c=(2,4,1))

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        p = relentless.Parallelepiped(a=(1,2,1),b=(3,4,5),c=(9,9,0))
        data = p.to_json()
        p_ = relentless.Parallelepiped.from_json(data)
        self.assertIsInstance(p_, relentless.Parallelepiped)
        np.testing.assert_allclose(p.a, p_.a)
        np.testing.assert_allclose(p.b, p_.b)
        np.testing.assert_allclose(p.c, p_.c)
        self.assertAlmostEqual(p.volume, p_.volume)

class test_TriclinicBox(unittest.TestCase):
    """Unit tests for relentless.TriclinicBox"""

    def test_init(self):
        """Test creation from data."""
        #test valid construction, LAMMPS convention
        t = relentless.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,
                                    convention=relentless.TriclinicBox.Convention.LAMMPS)
        np.testing.assert_allclose(t.a, np.array([1,0,0]))
        np.testing.assert_allclose(t.b, np.array([1,2,0]))
        np.testing.assert_allclose(t.c, np.array([0.75,2.25,3]))
        self.assertAlmostEqual(t.volume, 6)

        #test valid construction, HOOMD convention
        t = relentless.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=0.5,xz=0.25,yz=0.75,
                                    convention=relentless.TriclinicBox.Convention.HOOMD)
        np.testing.assert_allclose(t.a, np.array([1,0,0]))
        np.testing.assert_allclose(t.b, np.array([1,2,0]))
        np.testing.assert_allclose(t.c, np.array([0.75,2.25,3]))
        self.assertAlmostEqual(t.volume, 6)

        #test invalid constructions
        with self.assertRaises(ValueError):
            t = relentless.TriclinicBox(Lx=1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,convention='LAMMPS')
        with self.assertRaises(ValueError):
            t = relentless.TriclinicBox(Lx=-1,Ly=2,Lz=3,xy=1,xz=0.75,yz=2.25,
                                        convention=relentless.TriclinicBox.Convention.LAMMPS)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        #test LAMMPS convention
        c = relentless.TriclinicBox(Lx=3,Ly=4,Lz=5,xy=2,xz=3,yz=4,
                                    convention=relentless.TriclinicBox.Convention.LAMMPS)
        data = c.to_json()
        c_ = relentless.TriclinicBox.from_json(data)
        self.assertIsInstance(c_, relentless.TriclinicBox)
        np.testing.assert_allclose(c.a, c_.a)
        np.testing.assert_allclose(c.b, c_.b)
        np.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

        #test HOOMD convention
        c = relentless.TriclinicBox(Lx=3,Ly=4,Lz=5,xy=2,xz=3,yz=4,
                                    convention=relentless.TriclinicBox.Convention.HOOMD)
        data = c.to_json()
        c_ = relentless.TriclinicBox.from_json(data)
        self.assertIsInstance(c_, relentless.TriclinicBox)
        np.testing.assert_allclose(c.a, c_.a)
        np.testing.assert_allclose(c.b, c_.b)
        np.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

class test_Cuboid(unittest.TestCase):
    """Unit tests for relentless.Cuboid"""

    def test_init(self):
        """Test creation from data."""
        #test valid construction
        c = relentless.Cuboid(Lx=3,Ly=4,Lz=5)
        np.testing.assert_allclose(c.a, np.array([3,0,0]))
        np.testing.assert_allclose(c.b, np.array([0,4,0]))
        np.testing.assert_allclose(c.c, np.array([0,0,5]))
        self.assertAlmostEqual(c.volume, 60)

        #test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.Cuboid(Lx=-3,Ly=4,Lz=5)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.Cuboid(Lx=3,Ly=4,Lz=5)
        data = c.to_json()
        c_ = relentless.Cuboid.from_json(data)
        self.assertIsInstance(c_, relentless.Cuboid)
        np.testing.assert_allclose(c.a, c_.a)
        np.testing.assert_allclose(c.b, c_.b)
        np.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

class test_Cube(unittest.TestCase):
    """Unit tests for relentless.Cube"""

    def test_init(self):
        """Test creation from data."""
        #test valid construction
        c = relentless.Cube(L=3)
        np.testing.assert_allclose(c.a, np.array([3,0,0]))
        np.testing.assert_allclose(c.b, np.array([0,3,0]))
        np.testing.assert_allclose(c.c, np.array([0,0,3]))
        self.assertAlmostEqual(c.volume, 27)

        #test invalid construction
        with self.assertRaises(ValueError):
            c = relentless.Cube(L=-1)

    def test_to_from_json(self):
        """Test to_json and from_json methods."""
        c = relentless.Cube(L=3)
        data = c.to_json()
        c_ = relentless.Cube.from_json(data)
        self.assertIsInstance(c_, relentless.Cube)
        np.testing.assert_allclose(c.a, c_.a)
        np.testing.assert_allclose(c.b, c_.b)
        np.testing.assert_allclose(c.c, c_.c)
        self.assertAlmostEqual(c.volume, c_.volume)

if __name__ == '__main__':
    unittest.main()
