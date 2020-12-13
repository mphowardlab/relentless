"""Unit tests for core.data module."""
import os
import shutil
import tempfile
import unittest

import relentless

class test_Directory(unittest.TestCase):
    def setUp(self):
        self.f = tempfile.TemporaryDirectory()
        self.real_f = os.path.realpath(self.f.name)

    """Unit tests for core.data.Directory."""
    def test_init(self):
        """Test basic creation and methods of Directory."""
        cwd = os.getcwd()

        #test creation with existing path
        d = relentless.Directory(self.f.name)
        self.assertEqual(d.path, self.real_f)
        self.assertEqual(d._start, [])
        #enter and exit
        with d:
            self.assertEqual(d._start, [cwd])
            self.assertEqual(os.getcwd(), self.real_f)
        self.assertEqual(d._start, [])

        #test creation with non-existent path (absolute)
        foo = os.path.join(self.f.name,'foo')
        real_foo = os.path.realpath(foo)
        d1 = relentless.Directory(foo)
        self.assertEqual(d1.path, real_foo)
        self.assertEqual(d1._start, [])
        #enter and exit
        with d1:
            self.assertEqual(d1._start, [cwd])
            self.assertEqual(os.getcwd(), real_foo)
        self.assertEqual(d1._start, [])

        #test creation with non-existent path (recursive)
        foobar = os.path.join(self.f.name,'bar','foobar')
        real_foobar = os.path.realpath(foobar)
        d2 = relentless.Directory(foobar)
        self.assertEqual(d2.path, real_foobar)
        self.assertEqual(d2._start, [])
        #enter and exit
        with d2:
            self.assertEqual(d2._start, [cwd])
            self.assertEqual(os.getcwd(), real_foobar)
        self.assertEqual(d2._start, [])

        #test creation with invalid directory path
        x = tempfile.NamedTemporaryFile(dir=self.f.name)
        with self.assertRaises(OSError):
            d3 = relentless.Directory(x.name)
        x.close()

        #test unsuccessfully changing directories (by redundancy)
        with d:
            self.assertEqual(d._start, [cwd])
            with d1:
                self.assertEqual(d1._start, [d.path])
                with d1:
                    self.assertEqual(d1._start, [d.path,d1.path])
                self.assertEqual(d1._start, [d.path])
                with d2:
                    self.assertEqual(d2._start, [d1.path])
                    with d2:
                        self.assertEqual(d2._start, [d1.path,d2.path])
                    self.assertEqual(d2._start, [d1.path])
                self.assertEqual(d2._start, [])
            self.assertEqual(d1._start, [])
        self.assertEqual(d._start, [])

        #test unsuccessful exit after successful enter
        with d2:
            self.assertEqual(d2._start, [cwd])
            with d1:
                self.assertEqual(d1._start, [d2.path])
                shutil.rmtree(foobar)
            self.assertEqual(d1._start, [])
            self.assertEqual(os.getcwd(), d1.path) #no exit

    def test_context(self):
        """Test context methods for Directory."""
        #create nested directory structure
        d = relentless.Directory(self.f.name)
        d1 = d.directory('foo')
        d2 = d.directory('bar')
        d3 = d1.directory(os.path.join('bar','foobar'))
        self.assertEqual(d.path, self.real_f)
        self.assertEqual(d1.path, os.path.join(self.real_f,'foo'))
        self.assertEqual(d2.path, os.path.join(self.real_f,'bar'))
        self.assertEqual(d3.path, os.path.join(self.real_f,'foo','bar','foobar'))

        #test creating files
        x = d.file('spam.txt')
        x1 = d1.file('ham.txt')
        x2 = d1.file('eggs.txt')
        x3 = d3.file('baz.txt')
        open(x,'w').close()
        open(x1,'w').close()
        open(x2,'w').close()
        open(x3,'w').close()
        self.assertEqual(os.path.abspath(x), os.path.join(d.path,'spam.txt'))
        self.assertEqual(os.path.abspath(x1), os.path.join(d1.path,'ham.txt'))
        self.assertEqual(os.path.abspath(x2), os.path.join(d1.path,'eggs.txt'))
        self.assertEqual(os.path.abspath(x3), os.path.join(d3.path,'baz.txt'))

        #test clearing directory structure
        #delete sub-directory
        self.assertCountEqual(os.listdir(d1.path), ['bar','ham.txt','eggs.txt'])
        self.assertCountEqual(os.listdir(d3.path), ['baz.txt'])
        d3.clear()
        self.assertCountEqual(os.listdir(d1.path), ['bar','ham.txt','eggs.txt'])
        self.assertCountEqual(os.listdir(d3.path), [])
        #delete parent directory
        self.assertCountEqual(os.listdir(d.path), ['foo','bar','spam.txt'])
        d.clear()
        self.assertCountEqual(os.listdir(d.path), [])


    def tearDown(self):
        self.f.cleanup()
        self.real_f = None
        del self.f

class test_Project(unittest.TestCase):
    def setUp(self):
        self.f = tempfile.TemporaryDirectory()
        self.real_f = os.path.realpath(self.f.name)

    """Unit tests for core.data.Project."""
    def test_init(self):
        """Test basic creation of Project."""
        with relentless.Directory(self.f.name):
            #unspecified workspace and scratch
            p = relentless.data.Project()
            self.assertIsInstance(p.workspace, relentless.Directory)
            self.assertEqual(p.workspace.path, os.path.join(self.real_f,'workspace'))
            self.assertEqual(p.scratch.path, os.path.join(self.real_f,'workspace','scratch'))

            #specified workspace, unspecified scratch
            p = relentless.Project(workspace=relentless.data.Directory(self.f.name))
            self.assertIsInstance(p.workspace, relentless.Directory)
            self.assertEqual(p.workspace.path, self.real_f)
            self.assertEqual(p.scratch.path, os.path.join(self.real_f,'scratch'))

            #unspecified workspace, specified scratch
            p = relentless.Project(scratch=relentless.data.Directory(self.f.name))
            self.assertIsInstance(p.workspace, relentless.Directory)
            self.assertEqual(p.workspace.path, os.path.join(self.real_f,'workspace'))
            self.assertEqual(p.scratch.path, self.real_f)

            #specified workspace and scratch (as strings)
            p = relentless.Project(workspace='wrksp',scratch='scr')
            self.assertIsInstance(p.workspace, relentless.Directory)
            self.assertEqual(p.workspace.path, os.path.join(self.real_f,'wrksp'))
            self.assertEqual(p.scratch.path, os.path.join(self.real_f,'scr'))

    def tearDown(self):
        self.f.cleanup()
        self.real_f = None
        del self.f

if __name__ == '__main__':
    unittest.main()
