"""Unit tests for core module."""
import unittest

import relentless

class test_Interpolator(unittest.TestCase):
    """Unit tests for core.Interpolator."""

    def test_init(self):
        """Test creation from data."""
        raise NotImplementedError()

    def test_call(self):
        """Test calls, both scalar and array."""
        raise NotImplementedError()

    def test_extrap(self):
        """Test extrapolation calls."""
        raise NotImplementedError()

class test_PairMatrix(unittest.TestCase):
    """Unit tests for core.PairMatrix."""

    def test_init(self):
        """Test construction with different list types."""
        raise NotImplementedError()

    def test_accessors(self):
        """Test get and set methods on pairs."""
        raise NotImplementedError()

    def test_iteration(self):
        """Test iteration on the matrix."""
        raise NotImplementedError()

class test_TypeDict(unittest.TestCase):
    """Unit tests for core.TypeDict."""

    def test_init(self):
        """Test construction with different list types."""
        raise NotImplementedError()

    def test_accessors(self):
        """Test get and set methods on types."""
        raise NotImplementedError()

    def test_iteration(self):
        """Test iteration on the dictionary."""
        raise NotImplementedError()

    def test_copy(self):
        """Test copying custom dict to standard dict."""
        raise NotImplementedError()

class test_Variable(unittest.TestCase):
    """Unit tests for core.Variable."""

    def test_init(self):
        """Test construction with different bounds."""
        raise NotImplementedError()

    def test_clamp(self):
        """Test methods for clamping values with bounds."""
        raise NotImplementedError()

    def test_value(self):
        """Test methods for setting values and checking bounds."""
        raise NotImplementedError()
