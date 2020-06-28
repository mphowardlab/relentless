"""Unit tests for environment module."""
import unittest

import os
import relentless

class test_Directory(unittest.TestCase):
    """Unit tests for environment.core.Directory."""
    def setUp(self):
    	# create temporary scratch space
        pass

    def test_basic(self):
        """Test basic creation and methods of Directory."""
        raise NotImplementedError()

    def test_context(self):
        """Test context methods for Directory."""
        raise NotImplementedError()

    def tearDown(self):
        # clean up temporary space
        pass

class test_Policy(unittest.TestCase):
    """Unit tests for environment.core.Policy."""
    def test_basic(self):
        """Test creation of Policy in different configurations."""
        raise NotImplementedError()

    def test_invalid(self):
        """Test for invalid policy values."""
        raise NotImplementedError()

class test_Environment(unittest.TestCase):
    """Unit tests for environment.core.Environment."""
    def setUp(self):
        # create temporary scratch space
        pass

    def test_basic(self):
        """Test basic creation and methods of Environment."""
        raise NotImplementedError()

    def test_call(self):
        """Test variations of call, first with mock=True and then mock=False."""
        raise NotImplementedError()

    def test_project(self):
        """Check that project and data directories."""
        raise NotImplementedError()

    def tearDown(self):
        # clean up temporary space
        pass

def test_generic(unittest.TestCase):
    """Unit tests for generic environments."""
    def test_OpenMPI(self):
        """Test environment for OpenMPI."""
        raise NotImplementedError()

    def test_SLURM(self):
        """Test environment for SLURM."""
        raise NotImplementedError()

def test_TACC(unittest.TestCase):
    """Unit tests for TACC environments."""
    def test_Lonestar5(self):
        """Test environment for TACC Lonestar5."""
        raise NotImplementedError()

    def test_Stampede2(self):
        """Test environment for TACC Stampede2."""
        raise NotImplementedError()
