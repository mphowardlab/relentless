"""Unit tests for collections module."""

import unittest

import relentless


class test_FixedKeyDict(unittest.TestCase):
    """Unit tests for relentless.collections.FixedKeyDict."""

    def test_init(self):
        """Test construction with different list keys."""
        keys = ("A", "B")

        # test construction with tuple input
        d = relentless.collections.FixedKeyDict(keys=("A", "B"))
        self.assertCountEqual(d.keys(), keys)
        self.assertEqual([d[k] for k in d.keys()], [None, None])

        # test construction with list input
        d = relentless.collections.FixedKeyDict(keys=["A", "B"])
        self.assertCountEqual(d.keys(), keys)
        self.assertEqual([d[k] for k in d.keys()], [None, None])

        # test construction with defined default input
        d = relentless.collections.FixedKeyDict(keys=("A", "B"), default=1.0)
        self.assertCountEqual(d.keys(), keys)
        self.assertEqual([d[k] for k in d.keys()], [1.0, 1.0])

        # test construction with single-key tuple input
        keys = ("A",)
        d = relentless.collections.FixedKeyDict(keys=("A",))
        self.assertCountEqual(d.keys(), keys)
        self.assertEqual([d[k] for k in d.keys()], [None])

    def test_accessors(self):
        """Test get and set methods on keys."""
        d = relentless.collections.FixedKeyDict(keys=("A", "B"))

        # test setting and getting values
        d["A"] = 1.0
        self.assertEqual([d[k] for k in d.keys()], [1.0, None])
        d["B"] = 1.0
        self.assertEqual([d[k] for k in d.keys()], [1.0, 1.0])

        # test re-setting and getting values
        d["A"] = 2.0
        self.assertEqual([d[k] for k in d.keys()], [2.0, 1.0])
        d["B"] = 1.5
        self.assertEqual([d[k] for k in d.keys()], [2.0, 1.5])

        # test getting invalid key
        with self.assertRaises(KeyError):
            d["C"]

    def test_update(self):
        """Test update method to get and set keys."""
        d = relentless.collections.FixedKeyDict(keys=("A", "B"))

        self.assertEqual([d[k] for k in d.keys()], [None, None])

        # test updating both keys
        d.update({"A": 1.0, "B": 2.0})  # using dict
        self.assertEqual([d[k] for k in d.keys()], [1.0, 2.0])

        d.update(A=1.5, B=2.5)  # using kwargs
        self.assertEqual([d[k] for k in d.keys()], [1.5, 2.5])

        # test updating only one key at a time
        d.update({"A": 1.1})  # using dict
        self.assertEqual([d[k] for k in d.keys()], [1.1, 2.5])

        d.update(B=2.2)  # using kwargs
        self.assertEqual([d[k] for k in d.keys()], [1.1, 2.2])

        # test using *args length > 1
        with self.assertRaises(TypeError):
            d.update({"A": 3.0}, {"B": 4.0})

        # test using both *args and **kwargs
        d.update({"A": 3.0, "B": 2.0}, B=2.2)
        self.assertEqual([d[k] for k in d.keys()], [3.0, 2.2])

        # test using invalid kwarg
        with self.assertRaises(KeyError):
            d.update(C=2.5)

    def test_clear(self):
        """Test clear method to reset keys to default."""
        # test clear with no default set
        d = relentless.collections.FixedKeyDict(keys=("A", "B"))
        self.assertEqual([d[k] for k in d.keys()], [None, None])
        d.update(A=2, B=3)
        self.assertEqual([d[k] for k in d.keys()], [2.0, 3.0])
        d.clear()
        self.assertEqual([d[k] for k in d.keys()], [None, None])

        # test clear with set default
        d = relentless.collections.FixedKeyDict(keys=("A", "B"), default=1.0)
        self.assertEqual([d[k] for k in d.keys()], [1.0, 1.0])
        d.update(A=2, B=3)
        self.assertEqual([d[k] for k in d.keys()], [2.0, 3.0])
        d.clear()
        self.assertEqual([d[k] for k in d.keys()], [1.0, 1.0])

    def test_iteration(self):
        """Test iteration on the dictionary."""
        d = relentless.collections.FixedKeyDict(keys=("A", "B"))

        # test iteration for setting values
        for k in d:
            d[k] = 1.0
        self.assertEqual([d[k] for k in d.keys()], [1.0, 1.0])

        # test manual re-setting of values
        d["A"] = 2.0
        self.assertEqual([d[k] for k in d.keys()], [2.0, 1.0])
        d["B"] = 1.5
        self.assertEqual([d[k] for k in d.keys()], [2.0, 1.5])

        # test iteration for re-setting values
        for k in d:
            d[k] = 3.0
        self.assertEqual([d[k] for k in d.keys()], [3.0, 3.0])

    def test_copy(self):
        """Test copying custom dict to standard dict."""
        d = relentless.collections.FixedKeyDict(keys=("A", "B"))

        # test copying for empty dict
        dict_var = {"A": None, "B": None}
        self.assertEqual(dict(d), dict_var)

        # test copying for partially filled dict
        dict_var = {"A": None, "B": 1.0}
        d["B"] = 1.0
        self.assertEqual(dict(d), dict_var)

        # test copying for full dict
        dict_var = {"A": 1.0, "B": 1.0}
        d["A"] = 1.0
        self.assertEqual(dict(d), dict_var)


class test_PairMatrix(unittest.TestCase):
    """Unit tests for relentless.collections.PairMatrix."""

    def test_init(self):
        """Test construction with different list types."""
        # test construction with tuple input
        pairs = (("A", "B"), ("B", "B"), ("A", "A"))
        m = relentless.collections.PairMatrix(("A", "B"))
        self.assertCountEqual(tuple(m.keys()), pairs)

        # test construction with list input
        m = relentless.collections.PairMatrix(["A", "B"])
        self.assertCountEqual(tuple(m.keys()), pairs)

        # test construction with single type tuple
        pairs = (("A", "A"),)
        m = relentless.collections.PairMatrix(("A",))
        self.assertCountEqual(tuple(m.keys()), pairs)

    def test_accessors(self):
        """Test get and set methods on pairs."""
        m = relentless.collections.PairMatrix(("A", "B"), default={})

        # test set and get for each pair type
        m["A", "A"]["energy"] = 1.0
        self.assertEqual(m["A", "A"]["energy"], 1.0)
        self.assertEqual(m["A", "B"], {})
        self.assertEqual(m["B", "B"], {})

        m["A", "B"]["energy"] = -1.0
        self.assertEqual(m["A", "A"]["energy"], 1.0)
        self.assertEqual(m["A", "B"]["energy"], -1.0)
        self.assertEqual(m["B", "B"], {})

        m["B", "B"]["energy"] = 1.0
        self.assertEqual(m["A", "A"]["energy"], 1.0)
        self.assertEqual(m["A", "B"]["energy"], -1.0)
        self.assertEqual(m["B", "B"]["energy"], 1.0)

        # test key order equality
        self.assertEqual(m["A", "B"], m["B", "A"])

        # test re-set and get
        m["A", "A"]["energy"] = 2.0
        self.assertEqual(m["A", "A"]["energy"], 2.0)
        self.assertEqual(m["A", "B"]["energy"], -1.0)
        self.assertEqual(m["B", "B"]["energy"], 1.0)

        m["A", "B"]["energy"] = -1.5
        self.assertEqual(m["A", "A"]["energy"], 2.0)
        self.assertEqual(m["A", "B"]["energy"], -1.5)
        self.assertEqual(m["B", "B"]["energy"], 1.0)

        m["B", "B"]["energy"] = 0.0
        self.assertEqual(m["A", "A"]["energy"], 2.0)
        self.assertEqual(m["A", "B"]["energy"], -1.5)
        self.assertEqual(m["B", "B"]["energy"], 0.0)

        # test setting multiple parameters and get
        m["A", "A"]["mass"] = 1.0
        self.assertEqual(m["A", "A"]["mass"], 1.0)
        self.assertEqual(m["A", "A"]["energy"], 2.0)
        self.assertEqual(m["A", "A"], {"energy": 2.0, "mass": 1.0})

        m["A", "B"]["mass"] = 3.0
        self.assertEqual(m["A", "B"]["mass"], 3.0)
        self.assertEqual(m["A", "B"]["energy"], -1.5)
        self.assertEqual(m["A", "B"], {"energy": -1.5, "mass": 3.0})

        m["B", "B"]["mass"] = 5.0
        self.assertEqual(m["B", "B"]["mass"], 5.0)
        self.assertEqual(m["B", "B"]["energy"], 0.0)
        self.assertEqual(m["B", "B"], {"energy": 0.0, "mass": 5.0})

        # test setting paramters for invalid keys
        with self.assertRaises(KeyError):
            m["C", "C"]
        with self.assertRaises(KeyError):
            m["A", "C"]

    def test_iteration(self):
        """Test iteration on the matrix."""
        m = relentless.collections.PairMatrix(("A", "B"), default={})

        # test iteration for initialization
        for pair in m:
            m[pair]["mass"] = 2.0
            m[pair]["energy"] = 1.0
        self.assertEqual(m["A", "B"], {"energy": 1.0, "mass": 2.0})
        self.assertEqual(m["A", "A"], {"energy": 1.0, "mass": 2.0})
        self.assertEqual(m["B", "B"], {"energy": 1.0, "mass": 2.0})

        # test resetting values manually
        m["A", "B"]["mass"] = 2.5
        m["A", "A"]["energy"] = 1.5
        self.assertEqual(m["A", "B"], {"energy": 1.0, "mass": 2.5})
        self.assertEqual(m["A", "A"], {"energy": 1.5, "mass": 2.0})
        self.assertEqual(m["B", "B"], {"energy": 1.0, "mass": 2.0})

        # test re-iteration for setting values
        for pair in m:
            m[pair]["energy"] = 3.0
        self.assertEqual(m["A", "B"], {"energy": 3.0, "mass": 2.5})
        self.assertEqual(m["A", "A"], {"energy": 3.0, "mass": 2.0})
        self.assertEqual(m["B", "B"], {"energy": 3.0, "mass": 2.0})


class test_DefaultDict(unittest.TestCase):
    """Unit tests for relentless.collections.DefaultDict"""

    def test_funcs(self):
        """Test functionalities."""
        # instantiation
        d = relentless.collections.DefaultDict(default=1.0)
        self.assertAlmostEqual(d.default, 1.0)
        self.assertAlmostEqual(d["A"], 1.0)
        self.assertAlmostEqual(d["B"], 1.0)
        self.assertEqual(len(d), 0)

        # set individually
        d["A"] = 2.0
        self.assertAlmostEqual(d.default, 1.0)
        self.assertAlmostEqual(d["A"], 2.0)
        self.assertAlmostEqual(d["B"], 1.0)
        self.assertEqual(len(d), 1)

        # delete
        del d["A"]
        self.assertAlmostEqual(d.default, 1.0)
        self.assertAlmostEqual(d["A"], 1.0)
        self.assertAlmostEqual(d["B"], 1.0)
        self.assertEqual(len(d), 0)

        # iterate
        for key in ("A", "B"):
            d[key] = 2.5
        self.assertAlmostEqual(d.default, 1.0)
        self.assertAlmostEqual(d["A"], 2.5)
        self.assertAlmostEqual(d["B"], 2.5)
        self.assertEqual(len(d), 2)


if __name__ == "__main__":
    unittest.main()
