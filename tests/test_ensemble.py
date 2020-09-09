"""Unit tests for ensemble module."""
import tempfile
import unittest

import numpy as np

import relentless

class test_RDF(unittest.TestCase):
    """Unit tests for relentless.RDF"""

    def test_init(self):
        """Test creation from data"""
        #test valid construction
        r = [1,2,3]
        g = [2,9,5]
        rdf = relentless.RDF(r=r, g=g)
        np.testing.assert_allclose(rdf.table, np.array([[1,2],
                                                        [2,9],
                                                        [3,5]]))

        #test invalid construction with r and g having different lengths
        r = [1,2,3,4]
        with self.assertRaises(ValueError):
            rdf = relentless.RDF(r=r, g=g)

        #test invalid construction with non-strictly-increasing r
        r = [1,3,2]
        with self.assertRaises(ValueError):
            rdf = relentless.RDF(r=r, g=g)

class test_Parallelepiped(unittest.TestCase):
    """Unit tests for relentless.Parallelepiped"""

    def test_init(self):
        """Test creation from data."""
        #test valid construction
        p = relentless.Parallelepiped(a=(1,2,1),b=(3,4,5),c=(9,9,0))
        np.testing.assert_allclose(p.a, np.array([1,2,1]))
        np.testing.assert_allclose(p.b, np.array([3,4,5]))
        np.testing.assert_allclose(p.c, np.array([9,9,0]))
        np.testing.assert_allclose(p.matrix, np.array([[1,3,9],
                                                       [2,4,9],
                                                       [1,5,0]]))
        self.assertAlmostEqual(p.volume, 36)

        #test invalid construction
        with self.assertRaises(TypeError):
            p = relentless.Parallelepiped(a=(1,2,1),b=(3,4,5),c=(9,9))

class test_Cuboid(unittest.TestCase):
    """Unit tests for relentless.Cuboid"""

    def test_init(self):
        """Test creation from data."""
        #test valid construction
        c = relentless.Cuboid(Lx=3,Ly=4,Lz=5)
        np.testing.assert_allclose(c.a, np.array([3,0,0]))
        np.testing.assert_allclose(c.b, np.array([0,4,0]))
        np.testing.assert_allclose(c.c, np.array([0,0,5]))
        np.testing.assert_allclose(c.matrix, np.array([[3,0,0],
                                                       [0,4,0],
                                                       [0,0,5]]))
        np.testing.assert_allclose(c.volume, 60)

        #test invalid construction
        with self.assertRaises(TypeError):
            c = relentless.Cuboid(Lx=(3,4),Ly=4,Lz=5)

class test_Cube(unittest.TestCase):
    """Unit tests for relentless.Cube"""

    def test_init(self):
        """Test creation from data."""
        #test valid construction
        c = relentless.Cube(L=3)
        np.testing.assert_allclose(c.a, np.array([3,0,0]))
        np.testing.assert_allclose(c.b, np.array([0,3,0]))
        np.testing.assert_allclose(c.c, np.array([0,0,3]))
        np.testing.assert_allclose(c.matrix, np.array([[3,0,0],
                                                       [0,3,0],
                                                       [0,0,3]]))
        np.testing.assert_allclose(c.volume, 27)

        #test invalid construction
        with self.assertRaises(TypeError):
            c = relentless.Cube(L=(3,4,5))

class test_Ensemble(unittest.TestCase):
    """Unit tests for relentless.Ensemble"""

    def test_init(self):
        """Test creation from data."""
        #P and mu set
        ens = relentless.Ensemble(T=10, P=2.0, mu={'A':0.1,'B':0.2})
        self.assertCountEqual(ens.types, ('A','B'))
        self.assertCountEqual(ens.rdf.pairs, (('A','B'),('A','A'),('B','B')))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 10)
        self.assertAlmostEqual(ens.P, 2.0)
        self.assertEqual(ens.V, None)
        self.assertDictEqual(ens.mu.todict(), {'A':0.1,'B':0.2})
        self.assertDictEqual(ens.N.todict(), {'A':None,'B':None})
        self.assertDictEqual(ens.constant, {'P':True,
                                            'V':False,
                                            'mu':{'A':True,'B':True},
                                            'N':{'A':False,'B':False}})
        self.assertAlmostEqual(ens.beta, 0.1)

        v_obj = relentless.Cube(L=3.0)

        #V and N set
        ens = relentless.Ensemble(T=20, V=v_obj, N={'A':1,'B':2})
        self.assertCountEqual(ens.types, ('A','B'))
        self.assertCountEqual(ens.rdf.pairs, (('A','B'),('A','A'),('B','B')))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 20)
        self.assertEqual(ens.P, None)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertDictEqual(ens.mu.todict(), {'A':None,'B':None})
        self.assertDictEqual(ens.N.todict(), {'A':1,'B':2})
        self.assertDictEqual(ens.constant, {'P':False,
                                            'V':True,
                                            'mu':{'A':False,'B':False},
                                            'N':{'A':True,'B':True}})
        self.assertAlmostEqual(ens.beta, 0.05)

        #mu and N used for one type each
        ens = relentless.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        self.assertCountEqual(ens.types, ('A','B'))
        self.assertCountEqual(ens.rdf.pairs, (('A','B'),('A','A'),('B','B')))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 100)
        self.assertEqual(ens.P, None)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertDictEqual(ens.mu.todict(), {'A':0.1,'B':None})
        self.assertDictEqual(ens.N.todict(), {'A':None,'B':2})
        self.assertDictEqual(ens.constant, {'P':False,
                                            'V':True,
                                            'mu':{'A':True,'B':False},
                                            'N':{'A':False,'B':True}})
        self.assertAlmostEqual(ens.beta, 0.01)

        #test creation with single type
        ens = relentless.Ensemble(T=100, V=v_obj, mu={'A':0.1})
        self.assertCountEqual(ens.types, ('A',))
        self.assertCountEqual(ens.rdf.pairs, (('A','A'),))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 100)
        self.assertEqual(ens.P, None)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertDictEqual(ens.mu.todict(), {'A':0.1})
        self.assertDictEqual(ens.N.todict(), {'A':None})
        self.assertDictEqual(ens.constant, {'P':False,
                                            'V':True,
                                            'mu':{'A':True},
                                            'N':{'A':False}})
        self.assertAlmostEqual(ens.beta, 0.01)

        #test setting rdf
        ens = relentless.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        r = [1,2,3]
        g_ab = [2,9,5]
        g_aa = [3,7,4]
        g_bb = [1,9,3]
        ens.rdf['A','B'] = relentless.RDF(r=r, g=g_ab)
        ens.rdf['A','A'] = relentless.RDF(r=r, g=g_aa)
        ens.rdf['B','B'] = relentless.RDF(r=r, g=g_bb)
        np.testing.assert_allclose(ens.rdf['A','B'].table, np.array([[1,2],
                                                                     [2,9],
                                                                     [3,5]]))
        np.testing.assert_allclose(ens.rdf['A','A'].table, np.array([[1,3],
                                                                     [2,7],
                                                                     [3,4]]))
        np.testing.assert_allclose(ens.rdf['B','B'].table, np.array([[1,1],
                                                                     [2,9],
                                                                     [3,3]]))

        #test invalid setting of neither P nor V
        with self.assertRaises(ValueError):
            ens = relentless.Ensemble(T=10, mu={'A':0.1,'B':0.2})

        #test invalid setting of neither mu nor N
        with self.assertRaises(ValueError):
            ens = relentless.Ensemble(T=10, P=2.0)

        #test invalid setting of both P and V
        with self.assertRaises(ValueError):
            ens = relentless.Ensemble(T=10, P=2.0, V=v_obj)

        #test invalid setting of both mu and N
        with self.assertRaises(ValueError):
            ens = relentless.Ensemble(T=10, mu={'A':0.1,'B':0.2}, N={'A':1,'B':2})

        #test setting N as a float
        with self.assertRaises(TypeError):
            ens = relentless.Ensemble(T=10, P=1.0, N={'A':1.0})

    def test_set_conjugates(self):
        """Test setting constant and conjugate/fluctuating parameter values."""
        v_obj = relentless.Cube(L=3.0)
        v_obj1 = relentless.Cube(L=4.0)

        #P and mu set
        ens = relentless.Ensemble(T=10, P=2.0, mu={'A':0.1,'B':0.2})
        self.assertAlmostEqual(ens.P, 2.0)
        self.assertEqual(ens.V, None)
        self.assertDictEqual(ens.mu.todict(), {'A':0.1,'B':0.2})
        self.assertDictEqual(ens.N.todict(), {'A':None,'B':None})
        #set conjugate values
        ens.V = v_obj
        self.assertAlmostEqual(ens.V.volume, 27.0)
        ens.N['A'] = 1
        ens.N['B'] = 2
        self.assertDictEqual(ens.N.todict(), {'A':1,'B':2})
        ens.P = 3.0
        self.assertAlmostEqual(ens.P, 3.0)
        ens.mu['A'] = 0.3
        ens.mu['B'] = 0.4
        self.assertAlmostEqual(ens.mu.todict(), {'A':0.3,'B':0.4})

        #V and N set
        ens = relentless.Ensemble(T=20, V=v_obj, N={'A':1,'B':2})
        self.assertEqual(ens.P, None)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertDictEqual(ens.mu.todict(), {'A':None,'B':None})
        self.assertDictEqual(ens.N.todict(), {'A':1,'B':2})
        #set constant values
        ens.P = 1.0
        self.assertAlmostEqual(ens.P, 1.0)
        ens.mu['A'] = 0.1
        ens.mu['B'] = 0.2
        self.assertDictEqual(ens.mu.todict(), {'A':0.1,'B':0.2})
        ens.V = v_obj1
        self.assertAlmostEqual(ens.V.volume, 64.0)
        ens.N['A'] = 1
        ens.N['B'] = 2
        self.assertDictEqual(ens.N.todict(), {'A':1,'B':2})

        #mu and N used for one type each
        ens = relentless.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        self.assertEqual(ens.P, None)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertDictEqual(ens.mu.todict(), {'A':0.1,'B':None})
        self.assertDictEqual(ens.N.todict(), {'A':None,'B':2})
        #set constant values
        ens.P = 1.0
        self.assertAlmostEqual(ens.P, 1.0)
        ens.V = v_obj1
        self.assertAlmostEqual(ens.V.volume, 64.0)
        ens.mu['A'] = 0.2
        ens.mu['B'] = 0.3
        self.assertDictEqual(ens.mu.todict(), {'A':0.2,'B':0.3})
        ens.N['A'] = 2
        ens.N['B'] = 3
        self.assertDictEqual(ens.N.todict(), {'A':2,'B':3})

    def test_clear(self):
        """Test clear method and modifying attribute values"""
        v_obj = relentless.Cube(L=1.0)

        #P and mu set
        ens = relentless.Ensemble(T=10, P=2.0, mu={'A':0.1,'B':0.2})
        ens.V = v_obj
        ens.N['A'] = 1
        ens.N['B'] = 2
        ens.clear()
        self.assertAlmostEqual(ens.P, 2.0)
        self.assertEqual(ens.V, None)
        self.assertAlmostEqual(ens.mu.todict(), {'A':0.1,'B':0.2})
        self.assertEqual(ens.N.todict(), {'A':None,'B':None})

        #V and N set
        ens = relentless.Ensemble(T=20, V=v_obj, N={'A':1,'B':2})
        ens.P = 1.0
        ens.mu['A'] = 0.2
        ens.mu['B'] = 0.3
        ens.clear()
        self.assertAlmostEqual(ens.P, None)
        self.assertAlmostEqual(ens.V.volume, 1.0)
        self.assertAlmostEqual(ens.mu.todict(), {'A':None,'B':None})
        self.assertEqual(ens.N.todict(), {'A':1,'B':2})

        #mu and N used for one type each
        ens = relentless.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        ens.P = 1.0
        ens.mu['B'] = 0.2
        ens.N['A'] = 1
        ens.clear()
        self.assertAlmostEqual(ens.P, None)
        self.assertAlmostEqual(ens.V.volume, 1.0)
        self.assertDictEqual(ens.mu.todict(), {'A':0.1,'B':None})
        self.assertDictEqual(ens.N.todict(), {'A':None,'B':2})

        #test resetting rdf
        ens = relentless.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        r = [1,2,3]
        g_ab = [2,9,5]
        g_aa = [3,7,4]
        g_bb = [1,9,3]
        ens.rdf['A','B'] = relentless.RDF(r=r, g=g_ab)
        ens.rdf['A','A'] = relentless.RDF(r=r, g=g_aa)
        ens.rdf['B','B'] = relentless.RDF(r=r, g=g_bb)
        ens.clear()
        self.assertDictEqual(ens.rdf['A','B'], {})
        self.assertDictEqual(ens.rdf['A','A'], {})
        self.assertDictEqual(ens.rdf['B','B'], {})

    def test_copy(self):
        """Test copy method"""
        v_obj = relentless.Cube(L=1.0)

        #P and mu set
        ens = relentless.Ensemble(T=10, P=2.0, mu={'A':0.1,'B':0.2})
        ens.V = v_obj
        ens.N['A'] = 1
        ens.N['B'] = 2
        ens_ = ens.copy()
        assert ens_ is not ens
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertIsInstance(ens_.V, relentless.Cube)
        self.assertAlmostEqual(ens.V.volume, ens_.V.volume)
        self.assertDictEqual(ens.mu.todict(), ens_.mu.todict())
        self.assertDictEqual(ens_.N.todict(), ens.N.todict())

        #V and N set
        ens = relentless.Ensemble(T=20, V=v_obj, N={'A':1,'B':2})
        ens.P = 1.0
        ens.mu['A'] = 0.1
        ens.mu['B'] = 0.2
        ens_ = ens.copy()
        assert ens_ is not ens
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertIsInstance(ens_.V, relentless.Cube)
        self.assertAlmostEqual(ens.V.volume, ens_.V.volume)
        self.assertDictEqual(ens.mu.todict(), ens_.mu.todict())
        self.assertDictEqual(ens_.N.todict(), ens.N.todict())

        #mu and N used for one type each
        ens = relentless.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        ens.P = 1.0
        ens.mu['B'] = 0.2
        ens.N['A'] = 1
        ens_ = ens.copy()
        assert ens_ is not ens
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertIsInstance(ens_.V, relentless.Cube)
        self.assertAlmostEqual(ens.V.volume, ens_.V.volume)
        self.assertDictEqual(ens.mu.todict(), ens_.mu.todict())
        self.assertDictEqual(ens_.N.todict(), ens.N.todict())

        #test copying rdf
        ens = relentless.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        r = [1,2,3]
        g_ab = [2,9,5]
        g_aa = [3,7,4]
        g_bb = [1,9,3]
        ens.rdf['A','B'] = relentless.RDF(r=r, g=g_ab)
        ens.rdf['A','A'] = relentless.RDF(r=r, g=g_aa)
        ens.rdf['B','B'] = relentless.RDF(r=r, g=g_bb)
        ens_ = ens.copy()
        assert ens_ is not ens
        np.testing.assert_allclose(ens.rdf['A','B'].table, ens_.rdf['A','B'].table)
        np.testing.assert_allclose(ens.rdf['A','A'].table, ens_.rdf['A','A'].table)
        np.testing.assert_allclose(ens.rdf['B','B'].table, ens_.rdf['B','B'].table)

    def test_save_load(self):
        """Test save and load methods"""
        pass

if __name__ == '__main__':
    unittest.main()
