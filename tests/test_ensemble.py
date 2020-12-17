"""Unit tests for ensemble module."""
import tempfile
import unittest

import numpy as np

import relentless

class test_RDF(unittest.TestCase):
    """Unit tests for relentless.ensemble.RDF"""

    def test_init(self):
        """Test creation from data"""
        #test valid construction
        r = [1,2,3]
        g = [2,9,5]
        rdf = relentless.ensemble.RDF(r=r, g=g)
        np.testing.assert_allclose(rdf.table, np.array([[1,2],
                                                        [2,9],
                                                        [3,5]]))

        #test interpolation
        np.testing.assert_allclose(rdf([2.5,3.5]), [8.375,5.0])

        #test invalid construction with r and g having different lengths
        r = [1,2,3,4]
        with self.assertRaises(ValueError):
            rdf = relentless.ensemble.RDF(r=r, g=g)

        #test invalid construction with non-strictly-increasing r
        r = [1,3,2]
        with self.assertRaises(ValueError):
            rdf = relentless.ensemble.RDF(r=r, g=g)

class test_Ensemble(unittest.TestCase):
    """Unit tests for relentless.ensemble.Ensemble"""

    def test_init(self):
        """Test creation from data."""
        #P and mu set
        ens = relentless.ensemble.Ensemble(T=10, P=2.0, mu={'A':0.1,'B':0.2})
        self.assertCountEqual(ens.types, ('A','B'))
        self.assertCountEqual(ens.rdf.pairs, (('A','B'),('A','A'),('B','B')))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 10)
        self.assertAlmostEqual(ens.P, 2.0)
        self.assertEqual(ens.V, None)
        self.assertDictEqual(ens.mu.todict(), {'A':0.1,'B':0.2})
        self.assertDictEqual(ens.N.todict(), {'A':None,'B':None})
        self.assertDictEqual(ens.constant, {'T': True,
                                            'P':True,
                                            'V':False,
                                            'mu':{'A':True,'B':True},
                                            'N':{'A':False,'B':False}})
        self.assertAlmostEqual(ens.beta, 0.1)

        #V and N set, non-default value of kB
        v_obj = relentless.volume.Cube(L=3.0)
        ens = relentless.ensemble.Ensemble(T=20, V=v_obj, N={'A':1,'B':2}, kB=2.0)
        self.assertCountEqual(ens.types, ('A','B'))
        self.assertCountEqual(ens.rdf.pairs, (('A','B'),('A','A'),('B','B')))
        self.assertAlmostEqual(ens.kB, 2.0)
        self.assertAlmostEqual(ens.T, 20)
        self.assertEqual(ens.P, None)
        self.assertIs(ens.V,v_obj)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertDictEqual(ens.mu.todict(), {'A':None,'B':None})
        self.assertDictEqual(ens.N.todict(), {'A':1,'B':2})
        self.assertDictEqual(ens.constant, {'T': True,
                                            'P':False,
                                            'V':True,
                                            'mu':{'A':False,'B':False},
                                            'N':{'A':True,'B':True}})
        self.assertAlmostEqual(ens.beta, 0.025)

        #mu and N used for one type each
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        self.assertCountEqual(ens.types, ('A','B'))
        self.assertCountEqual(ens.rdf.pairs, (('A','B'),('A','A'),('B','B')))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 100)
        self.assertEqual(ens.P, None)
        self.assertIs(ens.V,v_obj)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertDictEqual(ens.mu.todict(), {'A':0.1,'B':None})
        self.assertDictEqual(ens.N.todict(), {'A':None,'B':2})
        self.assertDictEqual(ens.constant, {'T': True,
                                            'P':False,
                                            'V':True,
                                            'mu':{'A':True,'B':False},
                                            'N':{'A':False,'B':True}})
        self.assertAlmostEqual(ens.beta, 0.01)

        #test creation with single type
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, mu={'A':0.1})
        self.assertCountEqual(ens.types, ('A',))
        self.assertCountEqual(ens.rdf.pairs, (('A','A'),))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 100)
        self.assertEqual(ens.P, None)
        self.assertIs(ens.V,v_obj)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertDictEqual(ens.mu.todict(), {'A':0.1})
        self.assertDictEqual(ens.N.todict(), {'A':None})
        self.assertDictEqual(ens.constant, {'T': True,
                                            'P':False,
                                            'V':True,
                                            'mu':{'A':True},
                                            'N':{'A':False}})
        self.assertAlmostEqual(ens.beta, 0.01)

        #test setting rdf
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        r = [1,2,3]
        ens.rdf['A','B'] = relentless.ensemble.RDF(r=r, g=[2,9,5])
        ens.rdf['A','A'] = relentless.ensemble.RDF(r=r, g=[3,7,4])
        ens.rdf['B','B'] = relentless.ensemble.RDF(r=r, g=[1,9,3])
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
            ens = relentless.ensemble.Ensemble(T=10, mu={'A':0.1,'B':0.2})

        #test invalid setting of neither mu nor N
        with self.assertRaises(ValueError):
            ens = relentless.ensemble.Ensemble(T=10, P=2.0)

        #test invalid setting of both P and V
        with self.assertRaises(ValueError):
            ens = relentless.ensemble.Ensemble(T=10, P=2.0, V=v_obj)

        #test invalid setting of both mu and N
        with self.assertRaises(ValueError):
            ens = relentless.ensemble.Ensemble(T=10, mu={'A':0.1,'B':0.2}, N={'A':1,'B':2})
        with self.assertRaises(ValueError):
            ens = relentless.ensemble.Ensemble(T=10, mu={'A':0.1,'B':0.2}, N={'B':2})

        #test setting N as a float
        with self.assertRaises(TypeError):
            ens = relentless.ensemble.Ensemble(T=10, P=1.0, N={'A':1.0})

    def test_set_params(self):
        """Test setting constant and fluctuating parameter values."""
        v_obj = relentless.volume.Cube(L=3.0)
        v_obj1 = relentless.volume.Cube(L=4.0)

        #NVT ensemble
        ens = relentless.ensemble.Ensemble(T=10, V=v_obj, N={'A':1,'B':2})
        self.assertEqual(ens.P, None)
        self.assertIs(ens.V,v_obj)
        self.assertAlmostEqual(ens.V.volume, 27.0)
        self.assertDictEqual(ens.mu.todict(), {'A':None,'B':None})
        self.assertDictEqual(ens.N.todict(), {'A':1,'B':2})

        #set constant values
        ens.V = v_obj1
        self.assertIs(ens.V,v_obj1)
        self.assertAlmostEqual(ens.V.volume, 64.0)
        ens.N['A'] = 2
        ens.N['B'] = 3
        self.assertDictEqual(ens.N.todict(), {'A':2,'B':3})

        #set fluctuating/conjugate values
        ens.P = 2.0
        self.assertAlmostEqual(ens.P, 2.0)
        ens.mu['A'] = 0.2
        ens.mu['B'] = 0.3
        self.assertDictEqual(ens.mu.todict(), {'A':0.2,'B':0.3})

        #test invalid setting of N, mu dicts directly
        with self.assertRaises(AttributeError):
            ens.N = {'A':3,'B':4}
        with self.assertRaises(AttributeError):
            ens.mu = {'A':0.3,'B':0.4}

    def test_aka(self):
        """Test checking alternative names of ensembles."""
        nvt = relentless.ensemble.Ensemble(T=1.0,V=relentless.volume.Cube(1.0),N={'A':1})
        self.assertTrue(nvt.aka("NVT"))
        self.assertTrue(nvt.aka("canonical"))
        self.assertFalse(nvt.aka("NPT"))
        self.assertFalse(nvt.aka("muVT"))
        with self.assertRaises(ValueError):
            nvt.aka("alchemical")

        npt = relentless.ensemble.Ensemble(T=1.0,P=0.0,N={'A':1})
        self.assertFalse(npt.aka("NVT"))
        self.assertTrue(npt.aka("NPT"))
        self.assertTrue(npt.aka("isothermal-isobaric"))
        self.assertFalse(npt.aka("muVT"))

        grand = relentless.ensemble.Ensemble(T=1.0,V=relentless.volume.Cube(1.0),mu={'A':0.0})
        self.assertFalse(grand.aka("NVT"))
        self.assertFalse(grand.aka("NPT"))
        self.assertTrue(grand.aka("muVT"))
        self.assertTrue(grand.aka("grand canonical"))

        semigrand = relentless.ensemble.Ensemble(T=1.0,V=relentless.volume.Cube(1.0),N={'A':1},mu={'B':0.0})
        self.assertFalse(semigrand.aka("NVT"))
        self.assertFalse(semigrand.aka("NPT"))
        self.assertFalse(semigrand.aka("muVT"))

    def test_clear(self):
        """Test clear method and modifying attribute values"""
        v_obj = relentless.volume.Cube(L=1.0)

        #P and mu set
        ens = relentless.ensemble.Ensemble(T=10, P=2.0, mu={'A':0.1,'B':0.2})
        ens.V = v_obj
        ens.N['A'] = 1
        ens.N['B'] = 2
        ens.clear()
        self.assertAlmostEqual(ens.P, 2.0)
        self.assertEqual(ens.V, None)
        self.assertAlmostEqual(ens.mu.todict(), {'A':0.1,'B':0.2})
        self.assertEqual(ens.N.todict(), {'A':None,'B':None})

        #V and N set
        ens = relentless.ensemble.Ensemble(T=20, V=v_obj, N={'A':1,'B':2})
        ens.P = 1.0
        ens.mu['A'] = 0.2
        ens.mu['B'] = 0.3
        ens.clear()
        self.assertAlmostEqual(ens.P, None)
        self.assertAlmostEqual(ens.V.volume, 1.0)
        self.assertAlmostEqual(ens.mu.todict(), {'A':None,'B':None})
        self.assertEqual(ens.N.todict(), {'A':1,'B':2})

        #mu and N used for one type each
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        ens.P = 1.0
        ens.mu['B'] = 0.2
        ens.N['A'] = 1
        ens.clear()
        self.assertAlmostEqual(ens.P, None)
        self.assertAlmostEqual(ens.V.volume, 1.0)
        self.assertDictEqual(ens.mu.todict(), {'A':0.1,'B':None})
        self.assertDictEqual(ens.N.todict(), {'A':None,'B':2})

        #test resetting rdf
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        r = [1,2,3]
        ens.rdf['A','B'] = relentless.ensemble.RDF(r=r, g=[2,9,5])
        ens.rdf['A','A'] = relentless.ensemble.RDF(r=r, g=[3,7,4])
        ens.rdf['B','B'] = relentless.ensemble.RDF(r=r, g=[1,9,3])
        ens.clear()
        self.assertEqual(ens.rdf['A','B'], None)
        self.assertEqual(ens.rdf['A','A'], None)
        self.assertEqual(ens.rdf['B','B'], None)

    def test_copy(self):
        """Test copy method"""
        v_obj = relentless.volume.Cube(L=1.0)

        #P and mu set
        ens = relentless.ensemble.Ensemble(T=10, P=2.0, mu={'A':0.1,'B':0.2})
        ens.V = v_obj
        ens.N['A'] = 1
        ens.N['B'] = 2
        ens_ = ens.copy()
        self.assertIsNot(ens, ens_)
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertIsInstance(ens_.V, relentless.volume.Cube)
        self.assertAlmostEqual(ens.V.volume, ens_.V.volume)
        self.assertDictEqual(ens.mu.todict(), ens_.mu.todict())
        self.assertDictEqual(ens_.N.todict(), ens.N.todict())

        #V and N set
        ens = relentless.ensemble.Ensemble(T=20, V=v_obj, N={'A':1,'B':2})
        ens.P = 1.0
        ens.mu['A'] = 0.1
        ens.mu['B'] = 0.2
        ens_ = ens.copy()
        self.assertIsNot(ens_, ens)
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertIsInstance(ens_.V, relentless.volume.Cube)
        self.assertAlmostEqual(ens.V.volume, ens_.V.volume)
        self.assertDictEqual(ens.mu.todict(), ens_.mu.todict())
        self.assertDictEqual(ens_.N.todict(), ens.N.todict())

        #mu and N used for one type each
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        ens.P = 1.0
        ens.mu['B'] = 0.2
        ens.N['A'] = 1
        ens_ = ens.copy()
        self.assertIsNot(ens_, ens)
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertIsInstance(ens_.V, relentless.volume.Cube)
        self.assertAlmostEqual(ens.V.volume, ens_.V.volume)
        self.assertDictEqual(ens.mu.todict(), ens_.mu.todict())
        self.assertDictEqual(ens_.N.todict(), ens.N.todict())

        #test copying rdf
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        r = [1,2,3]
        ens.rdf['A','B'] = relentless.ensemble.RDF(r=r, g=[2,9,5])
        ens.rdf['A','A'] = relentless.ensemble.RDF(r=r, g=[3,7,4])
        ens.rdf['B','B'] = relentless.ensemble.RDF(r=r, g=[1,9,3])
        ens_ = ens.copy()
        self.assertIsNot(ens_, ens)
        self.assertIsNot(ens_.rdf, ens.rdf)
        np.testing.assert_allclose(ens.rdf['A','B'].table, ens_.rdf['A','B'].table)
        np.testing.assert_allclose(ens.rdf['A','A'].table, ens_.rdf['A','A'].table)
        np.testing.assert_allclose(ens.rdf['B','B'].table, ens_.rdf['B','B'].table)

    def test_save_from_file(self):
        """Test save and from_file methods"""
        temp = tempfile.NamedTemporaryFile()

        v_obj = relentless.volume.Cube(L=1.0)

        #P and mu set
        ens = relentless.ensemble.Ensemble(T=10, P=2.0, mu={'A':0.1,'B':0.2})
        ens.V = v_obj
        ens.N['A'] = 1
        ens.N['B'] = 2
        ens.save(temp.name)
        ens_ = relentless.ensemble.Ensemble.from_file(temp.name)
        self.assertIsNot(ens_, ens)
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertIsInstance(ens_.V, relentless.volume.Cube)
        self.assertAlmostEqual(ens.V.volume, ens_.V.volume)
        self.assertDictEqual(ens.mu.todict(), ens_.mu.todict())
        self.assertDictEqual(ens_.N.todict(), ens.N.todict())
        self.assertDictEqual(ens_.constant, ens.constant)

        #V and N set
        ens = relentless.ensemble.Ensemble(T=20, V=v_obj, N={'A':1,'B':2})
        ens.P = 1.0
        ens.mu['A'] = 0.1
        ens.mu['B'] = 0.2
        ens.save(temp.name)
        ens_ = relentless.ensemble.Ensemble.from_file(temp.name)
        self.assertIsNot(ens_, ens)
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertIsInstance(ens_.V, relentless.volume.Cube)
        self.assertAlmostEqual(ens.V.volume, ens_.V.volume)
        self.assertDictEqual(ens.mu.todict(), ens_.mu.todict())
        self.assertDictEqual(ens_.N.todict(), ens.N.todict())
        self.assertDictEqual(ens_.constant, ens.constant)

        #mu and N used for one type each
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        ens.P = 1.0
        ens.mu['B'] = 0.2
        ens.N['A'] = 1
        ens.save(temp.name)
        ens_ = relentless.ensemble.Ensemble.from_file(temp.name)
        self.assertIsNot(ens_, ens)
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertIsInstance(ens_.V, relentless.volume.Cube)
        self.assertAlmostEqual(ens.V.volume, ens_.V.volume)
        self.assertDictEqual(ens.mu.todict(), ens_.mu.todict())
        self.assertDictEqual(ens_.N.todict(), ens.N.todict())
        self.assertDictEqual(ens_.constant, ens.constant)

        #test saving/constructing rdf
        ens = relentless.ensemble.Ensemble(T=100, V=v_obj, mu={'A':0.1}, N={'B':2})
        r = [1,2,3]
        ens.rdf['A','B'] = relentless.ensemble.RDF(r=r, g=[2,9,5])
        ens.rdf['A','A'] = relentless.ensemble.RDF(r=r, g=[3,7,4])
        ens.save(temp.name)
        ens_ = relentless.ensemble.Ensemble.from_file(temp.name)
        self.assertIsNot(ens_, ens)
        self.assertIsNot(ens_.rdf, ens.rdf)
        np.testing.assert_allclose(ens.rdf['A','B'].table, ens_.rdf['A','B'].table)
        np.testing.assert_allclose(ens.rdf['A','A'].table, ens_.rdf['A','A'].table)
        self.assertEqual(ens.rdf['B','B'], ens_.rdf['B','B'])

        temp.close()

if __name__ == '__main__':
    unittest.main()
