"""Unit tests for ensemble module."""
import tempfile
import unittest

import numpy

import relentless

class test_RDF(unittest.TestCase):
    """Unit tests for relentless.ensemble.RDF"""

    def test_init(self):
        """Test creation from data"""
        # test valid construction
        r = [1,2,3]
        g = [2,9,5]
        rdf = relentless.ensemble.RDF(r=r, g=g)
        numpy.testing.assert_allclose(rdf.table, numpy.array([[1,2],
                                                              [2,9],
                                                              [3,5]]))

        # test interpolation
        numpy.testing.assert_allclose(rdf([2.5,3.5]), [8.375,5.0])

        # test invalid construction with r and g having different lengths
        r = [1,2,3,4]
        with self.assertRaises(ValueError):
            rdf = relentless.ensemble.RDF(r=r, g=g)

        # test invalid construction with non-strictly-increasing r
        r = [1,3,2]
        with self.assertRaises(ValueError):
            rdf = relentless.ensemble.RDF(r=r, g=g)

class test_Ensemble(unittest.TestCase):
    """Unit tests for relentless.ensemble.Ensemble"""

    def test_init(self):
        """Test creation from data."""
        # P and N set
        ens = relentless.ensemble.Ensemble(T=10, P=2.0, N={'A':1,'B':2})
        self.assertCountEqual(ens.types, ('A','B'))
        self.assertCountEqual(ens.rdf.pairs, (('A','B'),('A','A'),('B','B')))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 10)
        self.assertAlmostEqual(ens.P, 2.0)
        self.assertEqual(ens.V, None)
        self.assertEqual(dict(ens.N), {'A':1,'B':2})
        self.assertAlmostEqual(ens.beta, 0.1)

        # V and N set, non-default value of kB
        v_obj = relentless.volume.Cube(L=3.0)
        ens = relentless.ensemble.Ensemble(T=20, V=v_obj, N={'A':1,'B':2}, kB=2.0)
        self.assertCountEqual(ens.types, ('A','B'))
        self.assertCountEqual(ens.rdf.pairs, (('A','B'),('A','A'),('B','B')))
        self.assertAlmostEqual(ens.kB, 2.0)
        self.assertAlmostEqual(ens.T, 20)
        self.assertEqual(ens.P, None)
        self.assertIs(ens.V,v_obj)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertEqual(dict(ens.N), {'A':1,'B':2})
        self.assertAlmostEqual(ens.beta, 0.025)

        # one N is None
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, N={'A':None, 'B':2})
        self.assertCountEqual(ens.types, ('A','B'))
        self.assertCountEqual(ens.rdf.pairs, (('A','B'),('A','A'),('B','B')))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 100)
        self.assertEqual(ens.P, None)
        self.assertIs(ens.V,v_obj)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertEqual(dict(ens.N), {'A':None,'B':2})
        self.assertAlmostEqual(ens.beta, 0.01)

        # test creation with single type
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, N={'A':10})
        self.assertCountEqual(ens.types, ('A',))
        self.assertCountEqual(ens.rdf.pairs, (('A','A'),))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 100)
        self.assertEqual(ens.P, None)
        self.assertIs(ens.V,v_obj)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertEqual(dict(ens.N), {'A':10})
        self.assertAlmostEqual(ens.beta, 0.01)

        # test setting rdf
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, N={'A':1,'B':2})
        r = [1,2,3]
        ens.rdf['A','B'] = relentless.ensemble.RDF(r=r, g=[2,9,5])
        ens.rdf['A','A'] = relentless.ensemble.RDF(r=r, g=[3,7,4])
        ens.rdf['B','B'] = relentless.ensemble.RDF(r=r, g=[1,9,3])
        numpy.testing.assert_allclose(ens.rdf['A','B'].table, numpy.array([[1,2],
                                                                           [2,9],
                                                                           [3,5]]))
        numpy.testing.assert_allclose(ens.rdf['A','A'].table, numpy.array([[1,3],
                                                                           [2,7],
                                                                           [3,4]]))
        numpy.testing.assert_allclose(ens.rdf['B','B'].table, numpy.array([[1,1],
                                                                           [2,9],
                                                                           [3,3]]))

        # test setting N as a float
        with self.assertRaises(TypeError):
            ens = relentless.ensemble.Ensemble(T=10, P=1.0, N={'A':1.0})
        # test setting N with non-string type
        with self.assertRaises(TypeError):
            ens = relentless.ensemble.Ensemble(T=10, P=1.0, N={1:2})

    def test_set_params(self):
        """Test setting constant and fluctuating parameter values."""
        v_obj = relentless.volume.Cube(L=3.0)
        v_obj1 = relentless.volume.Cube(L=4.0)

        # NVT ensemble
        ens = relentless.ensemble.Ensemble(T=10, V=v_obj, N={'A':1,'B':2})
        self.assertEqual(ens.P, None)
        self.assertIs(ens.V,v_obj)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertEqual(dict(ens.N), {'A':1,'B':2})

        # set values
        ens.V = v_obj1
        self.assertIs(ens.V,v_obj1)
        self.assertAlmostEqual(ens.V.volume, 64.0)
        ens.N['A'] = 2
        ens.N['B'] = 3
        self.assertEqual(dict(ens.N), {'A':2,'B':3})

        # set other values
        ens.P = 2.0
        self.assertAlmostEqual(ens.P, 2.0)

        # test invalid setting of N directly
        with self.assertRaises(AttributeError):
            ens.N = {'A':3,'B':4}

    def test_copy(self):
        """Test copy method"""
        v_obj = relentless.volume.Cube(L=1.0)

        # P and mu set
        ens = relentless.ensemble.Ensemble(T=10, P=2.0, V=v_obj, N={'A':1,'B':2})
        ens_ = ens.copy()
        self.assertIsNot(ens, ens_)
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertIsInstance(ens_.V, relentless.volume.Cube)
        self.assertAlmostEqual(ens.V.volume, ens_.V.volume)
        self.assertEqual(dict(ens_.N), dict(ens.N))

        # test copying rdf
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, N={'A':1,'B':2})
        r = [1,2,3]
        ens.rdf['A','B'] = relentless.ensemble.RDF(r=r, g=[2,9,5])
        ens.rdf['A','A'] = relentless.ensemble.RDF(r=r, g=[3,7,4])
        ens.rdf['B','B'] = relentless.ensemble.RDF(r=r, g=[1,9,3])
        ens_ = ens.copy()
        self.assertIsNot(ens_, ens)
        self.assertIsNot(ens_.rdf, ens.rdf)
        numpy.testing.assert_allclose(ens.rdf['A','B'].table, ens_.rdf['A','B'].table)
        numpy.testing.assert_allclose(ens.rdf['A','A'].table, ens_.rdf['A','A'].table)
        numpy.testing.assert_allclose(ens.rdf['B','B'].table, ens_.rdf['B','B'].table)

    def test_save_from_file(self):
        """Test save and from_file methods"""
        temp = tempfile.NamedTemporaryFile()

        v_obj = relentless.volume.Cube(L=1.0)
        ens = relentless.ensemble.Ensemble(T=20, V=v_obj, P=1.0, N={'A':1,'B':2})
        ens.save(temp.name)
        ens_ = relentless.ensemble.Ensemble.from_file(temp.name)
        self.assertIsNot(ens_, ens)
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertIsInstance(ens_.V, relentless.volume.Cube)
        self.assertAlmostEqual(ens.V.volume, ens_.V.volume)
        self.assertEqual(dict(ens_.N), dict(ens.N))

        # test saving/constructing rdf
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, N={'A':1,'B':2})
        r = [1,2,3]
        ens.rdf['A','B'] = relentless.ensemble.RDF(r=r, g=[2,9,5])
        ens.rdf['A','A'] = relentless.ensemble.RDF(r=r, g=[3,7,4])
        ens.save(temp.name)
        ens_ = relentless.ensemble.Ensemble.from_file(temp.name)
        self.assertIsNot(ens_, ens)
        self.assertIsNot(ens_.rdf, ens.rdf)
        numpy.testing.assert_allclose(ens.rdf['A','B'].table, ens_.rdf['A','B'].table)
        numpy.testing.assert_allclose(ens.rdf['A','A'].table, ens_.rdf['A','A'].table)
        self.assertEqual(ens.rdf['B','B'], ens_.rdf['B','B'])

        temp.close()

if __name__ == '__main__':
    unittest.main()
