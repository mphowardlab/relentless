"""Unit tests for ensemble module."""
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
        """Test creation from data"""
        #conjugates not set, P and mu set
        ens = relentless.Ensemble(types=('A','B'), T=10, P=2.0, mu={'A':0.1,'B':0.2})
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

        #conjugates not set, V and N set
        ens = relentless.Ensemble(types=('A','B'), T=20, V=3.0, N={'A':1,'B':2})
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

        #conjugates not set, mu and N used for one type each
        ens = relentless.Ensemble(types=('A','B'), T=100, V=3.0, mu={'A':0.1}, N={'B':2})
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

        #custom conjugates set, allowing both P and V
        ens = relentless.Ensemble(types=('A','B'), T=100, P=1.5, V=3.0, mu={'A':0.1}, N={'B':2}, conjugates=('mu_B','N_A'))
        self.assertAlmostEqual(ens.P, 1.5)
        self.assertAlmostEqual(ens.V, 3.0)
        self.assertDictEqual(ens.mu, {'A':0.1,'B':None})
        self.assertDictEqual(ens.N, {'A':None,'B':2})
        self.assertCountEqual(ens.conjugates, ('mu_B','N_A'))

        #custom conjugates set, allowing both mu and N
        ens = relentless.Ensemble(types=('A','B'), T=100, V=3.0, mu={'A':0.1,'B':0.2}, N={'A':1,'B':2}, conjugates=('P',))
        self.assertEqual(ens.P, None)
        self.assertAlmostEqual(ens.V, 3.0)
        self.assertDictEqual(ens.mu, {'A':0.1,'B':0.2})
        self.assertDictEqual(ens.N, {'A':1,'B':2})
        self.assertCountEqual(ens.conjugates, ('P',))

        #custom conjugates set, allowing P, V, mu, and N
        ens = relentless.Ensemble(types=('A','B'), T=100, P=1.5, V=3.0, mu={'A':0.1,'B':0.2}, N={'A':1,'B':2}, conjugates=('q',))
        self.assertEqual(ens.P, 1.5)
        self.assertAlmostEqual(ens.V, 3.0)
        self.assertDictEqual(ens.mu, {'A':0.1,'B':0.2})
        self.assertDictEqual(ens.N, {'A':1,'B':2})
        self.assertCountEqual(ens.conjugates, ('q',))

        #test creation with mu and N dicts containing types not in ensemble
        ens = relentless.Ensemble(types=('A','B'), T=100, V=3.0, mu={'A':0.1,'C':0.2}, N={'B':2,'D':3})
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
        ens = relentless.Ensemble(types=('A',), T=100, V=3.0, mu={'A':0.1})
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

        #test invalid setting of neither P nor V
        with self.assertRaises(ValueError):
            ens = relentless.Ensemble(types=('A','B'), T=10, mu={'A':0.1,'B':0.2})

        #test invalid setting of neither mu nor N
        with self.assertRaises(ValueError):
            ens = relentless.Ensemble(types=('A','B'), T=10, P=2.0)

        #conjugates not set, test invalid setting of both P and V
        with self.assertRaises(ValueError):
            ens = relentless.Ensemble(types=('A','B'), T=10, P=2.0, V=3.0)

        #conjugates not set, test invalid setting of both mu and N
        with self.assertRaises(ValueError):
            ens = relentless.Ensemble(types=('A','B'), T=10, mu={'A':0.1,'B':0.2}, N={'A':1,'B':2})

    def test_reset(self):
        """Test reset method and modifying attribute values"""
        pass

    def test_copy(self):
        """Test copy method"""
        pass

    def test_save(self):
        """Test save method"""
        pass

    def test_load(self):
        """Test load method"""
        pass

if __name__ == '__main__':
    unittest.main()
