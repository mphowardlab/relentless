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
        d = relentless.data.Directory(self.f.name)
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
        d1 = relentless.data.Directory(foo)
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
        d2 = relentless.data.Directory(foobar)
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
            d3 = relentless.data.Directory(x.name)
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
        d = relentless.data.Directory(self.f.name)
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
        d3.clear_contents()
        self.assertCountEqual(os.listdir(d1.path), ['bar','ham.txt','eggs.txt'])
        self.assertCountEqual(os.listdir(d3.path), [])
        #delete parent directory
        self.assertCountEqual(os.listdir(d.path), ['foo','bar','spam.txt'])
        d.clear_contents()
        self.assertCountEqual(os.listdir(d.path), [])

    def test_path(self):
        # set path by constructor
        foo = os.path.join(self.f.name,'foo')
        real_foo = os.path.realpath(foo)
        d = relentless.data.Directory(foo)
        self.assertEqual(d.path, real_foo)
        self.assertTrue(os.path.exists(foo))

        # set path by property
        bar = os.path.join(self.f.name,'bar')
        real_bar = os.path.realpath(bar)
        d.path = bar
        self.assertEqual(d.path, real_bar)
        self.assertTrue(os.path.exists(bar))

        # cannot set path in context
        with d:
            with self.assertRaises(OSError):
                d.path = foo

    def test_move_contents(self):
        foo = os.path.join(self.f.name,'foo')
        bar = os.path.join(self.f.name,'bar')
        baz = os.path.join(self.f.name,'baz')

        dfoo = relentless.data.Directory(foo)
        dbar = relentless.data.Directory(bar)

        # create a file and directory in foo, it is not yet in bar
        open(dfoo.file('spam.txt'),'w').close()
        dfoo.directory('fizz')
        self.assertTrue(os.path.isfile(os.path.join(foo,'spam.txt')))
        self.assertTrue(os.path.isdir(os.path.join(foo,'fizz')))
        self.assertFalse(os.path.isfile(os.path.join(bar,'spam.txt')))
        self.assertFalse(os.path.isdir(os.path.join(bar,'fizz')))

        # move to bar
        dfoo.move_contents(dbar)
        self.assertFalse(os.path.isfile(os.path.join(foo,'spam.txt')))
        self.assertFalse(os.path.isdir(os.path.join(foo,'fizz')))
        self.assertTrue(os.path.isfile(os.path.join(bar,'spam.txt')))
        self.assertTrue(os.path.isdir(os.path.join(bar,'fizz')))

        # move to baz (doesn't exist as Directory yet)
        dbar.move_contents(baz)
        self.assertFalse(os.path.isfile(os.path.join(foo,'spam.txt')))
        self.assertFalse(os.path.isdir(os.path.join(foo,'fizz')))
        self.assertFalse(os.path.isfile(os.path.join(bar,'spam.txt')))
        self.assertFalse(os.path.isdir(os.path.join(bar,'fizz')))
        self.assertTrue(os.path.isfile(os.path.join(baz,'spam.txt')))
        self.assertTrue(os.path.isdir(os.path.join(baz,'fizz')))

    def test_copy_contents(self):
        foo = os.path.join(self.f.name,'foo')
        bar = os.path.join(self.f.name,'bar')
        baz = os.path.join(self.f.name,'baz')

        dfoo = relentless.data.Directory(foo)
        dbar = relentless.data.Directory(bar)

        # create a file and directory in foo, it is not yet in bar
        open(dfoo.file('spam.txt'),'w').close()
        dfoo.directory('fizz')
        self.assertTrue(os.path.isfile(os.path.join(foo,'spam.txt')))
        self.assertTrue(os.path.isdir(os.path.join(foo,'fizz')))
        self.assertFalse(os.path.isfile(os.path.join(bar,'spam.txt')))
        self.assertFalse(os.path.isdir(os.path.join(bar,'fizz')))

        # copy to bar
        dfoo.copy_contents(dbar)
        self.assertTrue(os.path.isfile(os.path.join(foo,'spam.txt')))
        self.assertTrue(os.path.isdir(os.path.join(foo,'fizz')))
        self.assertTrue(os.path.isfile(os.path.join(bar,'spam.txt')))
        self.assertTrue(os.path.isdir(os.path.join(bar,'fizz')))

        # copy to baz (doesn't exist as Directory yet)
        dfoo.copy_contents(baz)
        self.assertTrue(os.path.isfile(os.path.join(foo,'spam.txt')))
        self.assertTrue(os.path.isdir(os.path.join(foo,'fizz')))
        self.assertTrue(os.path.isfile(os.path.join(bar,'spam.txt')))
        self.assertTrue(os.path.isdir(os.path.join(bar,'fizz')))
        self.assertTrue(os.path.isfile(os.path.join(baz,'spam.txt')))
        self.assertTrue(os.path.isdir(os.path.join(baz,'fizz')))

    def tearDown(self):
        self.f.cleanup()
        self.real_f = None
        del self.f

if __name__ == '__main__':
    unittest.main()
