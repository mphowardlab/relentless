"""Unit tests for core.data module."""
import os
import shutil
import tempfile
import unittest

import relentless

class test_Directory(unittest.TestCase):
    """Unit tests for core.data.Directory."""
    def test_init(self):
        """Test basic creation and methods of Directory."""
        f = tempfile.TemporaryDirectory()

        cwd = os.getcwd()

        #test creation with existing path
        d = relentless.data.Directory(f.name)
        self.assertEqual(d.path, f.name)
        self.assertEqual(d._start, None)
        #enter and exit
        with d:
            self.assertEqual(d._start, cwd)
        self.assertEqual(d._start, None)

        #test creation with non-existent path (absolute)
        foo = f.name + '/foo'
        d1 = relentless.data.Directory(foo)
        self.assertEqual(d1.path, foo)
        self.assertEqual(d1._start, None)
        #enter and exit
        with d1:
            self.assertEqual(d1._start, cwd)
        self.assertEqual(d1._start, None)

        #test creation with non-existent path (recursive)
        foobar = foo + '/bar/foobar'
        d2 = relentless.data.Directory(foobar)
        self.assertEqual(d2.path, foobar)
        self.assertEqual(d2._start, None)
        #enter and exit
        with d2:
            self.assertEqual(d2._start, cwd)
        self.assertEqual(d2._start, None)

        #test creation with invalid directory path
        x = tempfile.NamedTemporaryFile(dir=f.name)
        with self.assertRaises(OSError):
            d3 = relentless.data.Directory(x.name)
        x.close()

        #test unsuccessfully changing directories (by redundancy)
        with d:
            self.assertEqual(d._start, cwd)
            with d1:
                self.assertEqual(d1._start, d.path)
                with d1:
                    self.assertEqual(d1._start, d.path) #_start is unchanged
                self.assertEqual(d1._start, None)
                with d2:
                    self.assertEqual(d2._start, d.path)
                self.assertEqual(d2._start, None)
            self.assertEqual(d1._start, None)
        self.assertEqual(d._start, None)

        #test unsuccessful exit after successful enter
        with d2:
            self.assertEqual(d2._start, cwd)
            with d1:
                self.assertEqual(d1._start, d2.path)
                shutil.rmtree(foobar)
            self.assertEqual(d1._start, None)
            self.assertEqual(os.getcwd(), d1.path) #no exit

        f.cleanup()

    def test_context(self):
        """Test context methods for Directory."""
        f = tempfile.TemporaryDirectory()

        #create nested directory structure
        d = relentless.data.Directory(f.name)
        d1 = d.directory('foo')
        d2 = d.directory('bar')
        d3 = d1.directory('bar/foobar')
        self.assertEqual(d.path, f.name)
        self.assertEqual(d1.path, f.name+'/foo')
        self.assertEqual(d2.path, f.name+'/bar')
        self.assertEqual(d3.path, f.name+'/foo/bar/foobar')

        #test creating files
        x = d.file('spam.txt')
        x1 = d1.file('ham.txt')
        x2 = d1.file('eggs.txt')
        x3 = d3.file('baz.txt')
        o = open(x, 'w')
        o.close()
        o1 = open(x1, 'w')
        o1.close()
        o2 = open(x2, 'w')
        o2.close()
        o3 = open(x3, 'w')
        o3.close()
        self.assertEqual(os.path.abspath(x), f.name+'/spam.txt')
        self.assertEqual(os.path.abspath(x1), f.name+'/foo/ham.txt')
        self.assertEqual(os.path.abspath(x2), f.name+'/foo/eggs.txt')
        self.assertEqual(os.path.abspath(x3), f.name+'/foo/bar/foobar/baz.txt')

        #test clearing directory structure
        #delete sub-directory
        self.assertCountEqual(os.listdir(f.name+'/foo'), ['bar','ham.txt','eggs.txt'])
        self.assertCountEqual(os.listdir(f.name+'/foo/bar/foobar'), ['baz.txt'])
        d3.clear()
        self.assertCountEqual(os.listdir(f.name+'/foo'), ['bar','ham.txt','eggs.txt'])
        self.assertCountEqual(os.listdir(f.name+'/foo/bar/foobar'), [])
        #delete parent directory
        self.assertCountEqual(os.listdir(f.name), ['foo','bar','spam.txt'])
        d.clear()
        self.assertCountEqual(os.listdir(f.name), [])

        f.cleanup()

class test_Project(unittest.TestCase):
    """Unit tests for core.data.Project."""
    def test_init(self):
        """Test basic creation of Project."""
        f = tempfile.TemporaryDirectory()
        os.chdir(f.name)

        #unspecified workspace and scratch
        p = relentless.data.Project()
        self.assertEqual(p.workspace.path, f.name+'/workspace')
        self.assertEqual(p.scratch.path, f.name+'/workspace/scratch')

        #specified workspace, unspecified scratch
        p = relentless.data.Project(workspace=relentless.data.Directory(f.name))
        self.assertEqual(p.workspace.path, f.name)
        self.assertEqual(p.scratch.path, f.name+'/scratch')

        #unspecified workspace, specified scratch
        p = relentless.data.Project(scratch=relentless.data.Directory(f.name))
        self.assertEqual(p.workspace.path, f.name+'/workspace')
        self.assertEqual(p.scratch.path, f.name)

        #specified workspace and scratch (as strings)
        p = relentless.data.Project(workspace='wrksp',scratch='scr')
        self.assertEqual(p.workspace.path, f.name+'/wrksp')
        self.assertEqual(p.scratch.path, f.name+'/scr')

        f.cleanup()

if __name__ == '__main__':
    unittest.main()
