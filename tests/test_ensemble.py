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
        self.assertDictEqual(ens.mu, {'A':0.1,'B':0.2})
        self.assertDictEqual(ens.N, {'A':None,'B':None})
        self.assertCountEqual(ens.conjugates, ('V','N_A','N_B'))
        self.assertAlmostEqual(ens.beta, 0.1)

        #V and N set
        ens = relentless.Ensemble(T=20, V=3.0, N={'A':1,'B':2})
        self.assertCountEqual(ens.types, ('A','B'))
        self.assertCountEqual(ens.rdf.pairs, (('A','B'),('A','A'),('B','B')))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 20)
        self.assertEqual(ens.P, None)
        self.assertAlmostEqual(ens.V, 3.0)
        self.assertDictEqual(ens.mu, {'A':None,'B':None})
        self.assertDictEqual(ens.N, {'A':1,'B':2})
        self.assertCountEqual(ens.conjugates, ('P','mu_A','mu_B'))
        self.assertAlmostEqual(ens.beta, 0.05)

        #mu and N used for one type each
        ens = relentless.Ensemble(T=100, V=3.0, mu={'A':0.1}, N={'B':2})
        self.assertCountEqual(ens.types, ('A','B'))
        self.assertCountEqual(ens.rdf.pairs, (('A','B'),('A','A'),('B','B')))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 100)
        self.assertEqual(ens.P, None)
        self.assertAlmostEqual(ens.V, 3.0)
        self.assertDictEqual(ens.mu, {'A':0.1,'B':None})
        self.assertDictEqual(ens.N, {'A':None,'B':2})
        self.assertCountEqual(ens.conjugates, ('P','mu_B','N_A'))
        self.assertAlmostEqual(ens.beta, 0.01)

        #test creation with single type
        ens = relentless.Ensemble(T=100, V=3.0, mu={'A':0.1})
        self.assertCountEqual(ens.types, ('A',))
        self.assertCountEqual(ens.rdf.pairs, (('A','A'),))
        self.assertAlmostEqual(ens.kB, 1.0)
        self.assertAlmostEqual(ens.T, 100)
        self.assertEqual(ens.P, None)
        self.assertAlmostEqual(ens.V, 3.0)
        self.assertDictEqual(ens.mu, {'A':0.1})
        self.assertDictEqual(ens.N, {'A':None})
        self.assertCountEqual(ens.conjugates, ('P','N_A'))
        self.assertAlmostEqual(ens.beta, 0.01)

        #test setting rdf
        ens = relentless.Ensemble(T=100, V=3.0, mu={'A':0.1}, N={'B':2})
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
            ens = relentless.Ensemble(T=10, P=2.0, V=3.0)

        #test invalid setting of both mu and N
        with self.assertRaises(ValueError):
            ens = relentless.Ensemble(T=10, mu={'A':0.1,'B':0.2}, N={'A':1,'B':2})

        #test setting N as a float
        with self.assertRaises(TypeError):
            ens = relentless.Ensemble(T=10, P=1.0, N={'A':1.0})

    def test_set_constants(self):
        """Test setting constant parameter values."""
        #P and mu set
        ens = relentless.Ensemble(T=10, P=2.0, mu={'A':0.1,'B':0.2})
        self.assertAlmostEqual(ens.P, 2.0)
        self.assertEqual(ens.V, None)
        self.assertDictEqual(ens.mu, {'A':0.1,'B':0.2})
        self.assertDictEqual(ens.N, {'A':None,'B':None})
        #set constant values
        ens.V = 1.0
        self.assertAlmostEqual(ens.V, 1.0)
        ens.N = {'A':1,'B':2}
        self.assertDictEqual(ens.N, {'A':1,'B':2})
        ens.P = 3.0
        self.assertAlmostEqual(ens.P, 3.0)
        ens.mu = {'A':0.3,'B':0.4}
        self.assertAlmostEqual(ens.mu, {'A':0.3,'B':0.4})

        #V and N set
        ens = relentless.Ensemble(T=20, V=3.0, N={'A':1,'B':2})
        self.assertEqual(ens.P, None)
        self.assertAlmostEqual(ens.V, 3.0)
        self.assertDictEqual(ens.mu, {'A':None,'B':None})
        self.assertDictEqual(ens.N, {'A':1,'B':2})
        #set constant values
        ens.P = 1.0
        self.assertAlmostEqual(ens.P, 1.0)
        ens.mu = {'A':0.1,'B':0.2}
        self.assertDictEqual(ens.mu, {'A':0.1,'B':0.2})
        ens.V = 4.0
        self.assertAlmostEqual(ens.V, 4.0)
        ens.N = {'A':1,'B':2}
        self.assertDictEqual(ens.N, {'A':1,'B':2})

        #mu and N used for one type each
        ens = relentless.Ensemble(T=100, V=3.0, mu={'A':0.1}, N={'B':2})
        self.assertEqual(ens.P, None)
        self.assertAlmostEqual(ens.V, 3.0)
        self.assertDictEqual(ens.mu, {'A':0.1,'B':None})
        self.assertDictEqual(ens.N, {'A':None,'B':2})
        #set constant values
        ens.P = 1.0
        self.assertAlmostEqual(ens.P, 1.0)
        ens.V = 4.0
        self.assertAlmostEqual(ens.V, 4.0)
        ens.mu = {'A':0.2,'B':0.3}
        self.assertDictEqual(ens.mu, {'A':0.2,'B':0.3})
        ens.N = {'A':2,'B':3}
        self.assertDictEqual(ens.N, {'A':2,'B':3})

    def test_reset(self):
        """Test reset method and modifying attribute values"""
        #P and mu set
        ens = relentless.Ensemble(T=10, P=2.0, mu={'A':0.1,'B':0.2})
        ens.V = 1.0
        ens.N = {'A':1,'B':2}
        ens.reset()
        self.assertAlmostEqual(ens.P, 2.0)
        self.assertEqual(ens.V, None)
        self.assertAlmostEqual(ens.mu, {'A':0.1,'B':0.2})
        self.assertEqual(ens.N, {'A':None,'B':None})

        #V and N set
        ens = relentless.Ensemble(T=20, V=3.0, N={'A':1,'B':2})
        ens.P = 1.0
        ens.mu = {'A':0.1,'B':0.2}
        ens.reset()
        self.assertAlmostEqual(ens.P, None)
        self.assertEqual(ens.V, 3.0)
        self.assertAlmostEqual(ens.mu, {'A':None,'B':None})
        self.assertEqual(ens.N, {'A':1,'B':2})

        #mu and N used for one type each
        ens = relentless.Ensemble(T=100, V=3.0, mu={'A':0.1}, N={'B':2})
        ens.P = 1.0
        ens.mu['B'] = 0.2
        ens.N['A'] = 1
        ens.reset()
        self.assertAlmostEqual(ens.P, None)
        self.assertEqual(ens.V, 3.0)
        self.assertAlmostEqual(ens.mu, {'A':0.1,'B':None})
        self.assertEqual(ens.N, {'A':None,'B':2})

        #test resetting rdf
        ens = relentless.Ensemble(T=100, V=3.0, mu={'A':0.1}, N={'B':2})
        r = [1,2,3]
        g_ab = [2,9,5]
        g_aa = [3,7,4]
        g_bb = [1,9,3]
        ens.rdf['A','B'] = relentless.RDF(r=r, g=g_ab)
        ens.rdf['A','A'] = relentless.RDF(r=r, g=g_aa)
        ens.rdf['B','B'] = relentless.RDF(r=r, g=g_bb)
        ens.reset()
        self.assertDictEqual(ens.rdf['A','B'], {})
        self.assertDictEqual(ens.rdf['A','A'], {})
        self.assertDictEqual(ens.rdf['B','B'], {})

    def test_copy(self):
        """Test copy method"""
        #P and mu set
        ens = relentless.Ensemble(T=10, P=2.0, mu={'A':0.1,'B':0.2})
        ens.V = 1.0
        ens.N = {'A':1,'B':2}
        ens_ = ens.copy()
        assert ens_ is not ens
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.P, ens_.P)
        self.assertEqual(ens_.V, None)
        self.assertDictEqual(ens.mu, ens_.mu)
        self.assertEqual(ens_.V, None)
        self.assertDictEqual(ens_.N, {'A':None,'B':None})

        #V and N set
        ens = relentless.Ensemble(T=20, V=3.0, N={'A':1,'B':2})
        ens.P = 1.0
        ens.mu = {'A':0.1,'B':0.2}
        ens_ = ens.copy()
        assert ens_ is not ens
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.V, ens_.V)
        self.assertDictEqual(ens.N, ens_.N)
        self.assertEqual(ens_.P, None)
        self.assertAlmostEqual(ens_.mu, {'A':None,'B':None})

        #mu and N used for one type each
        ens = relentless.Ensemble(T=100, V=3.0, mu={'A':0.1}, N={'B':2})
        ens.P = 1.0
        ens.mu['B'] = 0.2
        ens.N['A'] = 1
        ens_ = ens.copy()
        assert ens_ is not ens
        self.assertCountEqual(ens.types, ens_.types)
        self.assertCountEqual(ens.rdf.pairs, ens_.rdf.pairs)
        self.assertAlmostEqual(ens.T, ens_.T)
        self.assertAlmostEqual(ens.V, ens_.V)
        self.assertEqual(ens_.P, None)
        self.assertDictEqual(ens_.mu, {'A':0.1,'B':None})
        self.assertDictEqual(ens_.N, {'A':None,'B':2})

        #test copying rdf
        ens = relentless.Ensemble(T=100, V=3.0, mu={'A':0.1}, N={'B':2})
        r = [1,2,3]
        g_ab = [2,9,5]
        g_aa = [3,7,4]
        g_bb = [1,9,3]
        ens.rdf['A','B'] = relentless.RDF(r=r, g=g_ab)
        ens.rdf['A','A'] = relentless.RDF(r=r, g=g_aa)
        ens.rdf['B','B'] = relentless.RDF(r=r, g=g_bb)
        ens_ = ens.copy()
        assert ens_ is not ens
        self.assertDictEqual(ens_.rdf['A','B'], {})
        self.assertDictEqual(ens_.rdf['A','A'], {})
        self.assertDictEqual(ens_.rdf['B','B'], {})
        np.testing.assert_allclose(ens.rdf['A','B'].table, np.array([[1,2],
                                                                     [2,9],
                                                                     [3,5]]))
        np.testing.assert_allclose(ens.rdf['A','A'].table, np.array([[1,3],
                                                                     [2,7],
                                                                     [3,4]]))
        np.testing.assert_allclose(ens.rdf['B','B'].table, np.array([[1,1],
                                                                     [2,9],
                                                                     [3,3]]))

    def test_save_load(self):
        """Test save and load methods"""
        pass

if __name__ == '__main__':
    unittest.main()
