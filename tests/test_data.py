"""Unit tests for core.data module."""

import os
import shutil
import tempfile
import unittest

import relentless


class test_Directory(unittest.TestCase):
    def setUp(self):
        if relentless.mpi.world.rank_is_root:
            self._tmp = tempfile.TemporaryDirectory()
            directory = self._tmp.name
        else:
            directory = None
        self.f = relentless.mpi.world.bcast(directory)
        self.real_f = os.path.realpath(self.f)
        relentless.mpi.world.barrier()

    """Unit tests for core.data.Directory."""

    def test_init(self):
        """Test basic creation and methods of Directory."""
        cwd = os.getcwd()

        # test creation with existing path
        d = relentless.data.Directory(self.f, create=relentless.mpi.world.rank_is_root)
        relentless.mpi.world.barrier()
        self.assertEqual(d.path, self.real_f)
        self.assertEqual(d._start, [])
        # enter and exit
        with d:
            self.assertEqual(d._start, [cwd])
            self.assertEqual(os.getcwd(), self.real_f)
        self.assertEqual(d._start, [])

        # test creation with non-existent path (absolute)
        foo = os.path.join(self.f, "foo")
        real_foo = os.path.realpath(foo)
        d1 = relentless.data.Directory(foo, create=relentless.mpi.world.rank_is_root)
        relentless.mpi.world.barrier()
        self.assertEqual(d1.path, real_foo)
        self.assertEqual(d1._start, [])
        # enter and exit
        with d1:
            self.assertEqual(d1._start, [cwd])
            self.assertEqual(os.getcwd(), real_foo)
        self.assertEqual(d1._start, [])

        # test creation with non-existent path (recursive)
        foobar = os.path.join(self.f, "bar", "foobar")
        real_foobar = os.path.realpath(foobar)
        d2 = relentless.data.Directory(foobar, create=relentless.mpi.world.rank_is_root)
        relentless.mpi.world.barrier()
        self.assertEqual(d2.path, real_foobar)
        self.assertEqual(d2._start, [])
        # enter and exit
        with d2:
            self.assertEqual(d2._start, [cwd])
            self.assertEqual(os.getcwd(), real_foobar)
        self.assertEqual(d2._start, [])

        # test unsuccessfully changing directories (by redundancy)
        with d:
            self.assertEqual(d._start, [cwd])
            with d1:
                self.assertEqual(d1._start, [d.path])
                with d1:
                    self.assertEqual(d1._start, [d.path, d1.path])
                self.assertEqual(d1._start, [d.path])
                with d2:
                    self.assertEqual(d2._start, [d1.path])
                    with d2:
                        self.assertEqual(d2._start, [d1.path, d2.path])
                    self.assertEqual(d2._start, [d1.path])
                self.assertEqual(d2._start, [])
            self.assertEqual(d1._start, [])
        self.assertEqual(d._start, [])

        # test unsuccessful exit after successful enter
        with d2:
            self.assertEqual(d2._start, [cwd])
            with d1:
                self.assertEqual(d1._start, [d2.path])
                if relentless.mpi.world.rank_is_root:
                    shutil.rmtree(foobar)
                relentless.mpi.world.barrier()
            self.assertEqual(d1._start, [])
            self.assertEqual(os.getcwd(), d1.path)  # no exit

    def test_context(self):
        """Test context methods for Directory."""
        # create nested directory structure
        d = relentless.data.Directory(self.f, create=relentless.mpi.world.rank_is_root)
        relentless.mpi.world.barrier()
        d1 = d.directory("foo", create=relentless.mpi.world.rank_is_root)
        d2 = d.directory("bar", create=relentless.mpi.world.rank_is_root)
        d3 = d1.directory(
            os.path.join("bar", "foobar"), create=relentless.mpi.world.rank_is_root
        )
        relentless.mpi.world.barrier()
        self.assertEqual(d.path, self.real_f)
        self.assertEqual(d1.path, os.path.join(self.real_f, "foo"))
        self.assertEqual(d2.path, os.path.join(self.real_f, "bar"))
        self.assertEqual(d3.path, os.path.join(self.real_f, "foo", "bar", "foobar"))

        # test creating files
        x = d.file("spam.txt")
        x1 = d1.file("ham.txt")
        x2 = d1.file("eggs.txt")
        x3 = d3.file("baz.txt")
        if relentless.mpi.world.rank_is_root:
            open(x, "w").close()
            open(x1, "w").close()
            open(x2, "w").close()
            open(x3, "w").close()
        relentless.mpi.world.barrier()
        self.assertEqual(os.path.abspath(x), os.path.join(d.path, "spam.txt"))
        self.assertEqual(os.path.abspath(x1), os.path.join(d1.path, "ham.txt"))
        self.assertEqual(os.path.abspath(x2), os.path.join(d1.path, "eggs.txt"))
        self.assertEqual(os.path.abspath(x3), os.path.join(d3.path, "baz.txt"))

        # test clearing directory structure
        # delete sub-directory
        self.assertCountEqual(os.listdir(d1.path), ["bar", "ham.txt", "eggs.txt"])
        self.assertCountEqual(os.listdir(d3.path), ["baz.txt"])
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            d3.clear_contents()
        relentless.mpi.world.barrier()
        self.assertCountEqual(os.listdir(d1.path), ["bar", "ham.txt", "eggs.txt"])
        self.assertCountEqual(os.listdir(d3.path), [])
        # delete parent directory
        self.assertCountEqual(os.listdir(d.path), ["foo", "bar", "spam.txt"])
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            d.clear_contents()
        relentless.mpi.world.barrier()
        self.assertCountEqual(os.listdir(d.path), [])

    def test_path(self):
        # set path by constructor
        foo = os.path.join(self.f, "foo")
        real_foo = os.path.realpath(foo)
        d = relentless.data.Directory(foo, create=relentless.mpi.world.rank_is_root)
        relentless.mpi.world.barrier()
        self.assertEqual(d.path, real_foo)
        self.assertTrue(os.path.exists(foo))

    def test_move_contents(self):
        foo = os.path.join(self.f, "foo")
        bar = os.path.join(self.f, "bar")
        baz = os.path.join(self.f, "baz")

        dfoo = relentless.data.Directory(foo, create=relentless.mpi.world.rank_is_root)
        dbar = relentless.data.Directory(bar, create=relentless.mpi.world.rank_is_root)
        relentless.mpi.world.barrier()

        # create a file and directory in foo, it is not yet in bar
        if relentless.mpi.world.rank_is_root:
            open(dfoo.file("spam.txt"), "w").close()
        dfoo.directory("fizz", create=relentless.mpi.world.rank_is_root)
        relentless.mpi.world.barrier()
        self.assertTrue(os.path.isfile(os.path.join(foo, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(foo, "fizz")))
        self.assertFalse(os.path.isfile(os.path.join(bar, "spam.txt")))
        self.assertFalse(os.path.isdir(os.path.join(bar, "fizz")))

        # move to bar
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            dfoo.move_contents(dbar)
        relentless.mpi.world.barrier()
        self.assertFalse(os.path.isfile(os.path.join(foo, "spam.txt")))
        self.assertFalse(os.path.isdir(os.path.join(foo, "fizz")))
        self.assertTrue(os.path.isfile(os.path.join(bar, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(bar, "fizz")))

        # move to baz (doesn't exist as Directory yet)
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            dbar.move_contents(baz)
        relentless.mpi.world.barrier()
        self.assertFalse(os.path.isfile(os.path.join(foo, "spam.txt")))
        self.assertFalse(os.path.isdir(os.path.join(foo, "fizz")))
        self.assertFalse(os.path.isfile(os.path.join(bar, "spam.txt")))
        self.assertFalse(os.path.isdir(os.path.join(bar, "fizz")))
        self.assertTrue(os.path.isfile(os.path.join(baz, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(baz, "fizz")))

        # create a copy of the file and directory currently in baz in foo
        if relentless.mpi.world.rank_is_root:
            open(dfoo.file("spam.txt"), "w").close()
        dfoo.directory("fizz", create=relentless.mpi.world.rank_is_root)
        relentless.mpi.world.barrier()
        self.assertTrue(os.path.isfile(os.path.join(foo, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(foo, "fizz")))
        self.assertTrue(os.path.isfile(os.path.join(baz, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(baz, "fizz")))

        # add one file to each fizz named the same and one named different
        foofizz = os.path.join(foo, "fizz")
        bazfizz = os.path.join(baz, "fizz")

        dfoofizz = relentless.data.Directory(
            foofizz, create=relentless.mpi.world.rank_is_root
        )
        dbazfizz = relentless.data.Directory(
            bazfizz, create=relentless.mpi.world.rank_is_root
        )

        if relentless.mpi.world.rank_is_root:
            open(dfoofizz.file("buzz.txt"), "w").close()
            open(dfoofizz.file("fred.txt"), "w").close()
            open(dbazfizz.file("buzz.txt"), "w").close()
            open(dbazfizz.file("jim.txt"), "w").close()
        relentless.mpi.world.barrier()

        # move file and directory from foo to baz
        if relentless.mpi.world.rank_is_root:
            dfoo.move_contents(baz)
        relentless.mpi.world.barrier()
        self.assertFalse(os.path.isfile(os.path.join(foo, "spam.txt")))
        self.assertFalse(os.path.isdir(os.path.join(foo, "fizz")))
        self.assertFalse(os.path.isdir(os.path.join(foofizz, "buzz.txt")))
        self.assertFalse(os.path.isdir(os.path.join(foofizz, "fred.txt")))
        self.assertTrue(os.path.isfile(os.path.join(baz, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(baz, "fizz")))
        self.assertTrue(os.path.isfile(os.path.join(bazfizz, "buzz.txt")))
        self.assertTrue(os.path.isfile(os.path.join(bazfizz, "fred.txt")))
        self.assertTrue(os.path.isfile(os.path.join(bazfizz, "jim.txt")))

    def test_copy_contents(self):
        foo = os.path.join(self.f, "foo")
        bar = os.path.join(self.f, "bar")
        baz = os.path.join(self.f, "baz")

        dfoo = relentless.data.Directory(foo, create=relentless.mpi.world.rank_is_root)
        dbar = relentless.data.Directory(bar, create=relentless.mpi.world.rank_is_root)
        relentless.mpi.world.barrier()

        # create a file and directory in foo, it is not yet in bar
        if relentless.mpi.world.rank_is_root:
            open(dfoo.file("spam.txt"), "w").close()
        dfoo.directory("fizz", create=relentless.mpi.world.rank_is_root)
        relentless.mpi.world.barrier()
        self.assertTrue(os.path.isfile(os.path.join(foo, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(foo, "fizz")))
        self.assertFalse(os.path.isfile(os.path.join(bar, "spam.txt")))
        self.assertFalse(os.path.isdir(os.path.join(bar, "fizz")))

        # copy to bar
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            dfoo.copy_contents(dbar)
        relentless.mpi.world.barrier()
        self.assertTrue(os.path.isfile(os.path.join(foo, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(foo, "fizz")))
        self.assertTrue(os.path.isfile(os.path.join(bar, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(bar, "fizz")))

        # copy to baz (doesn't exist as Directory yet)
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            dfoo.copy_contents(baz)
        relentless.mpi.world.barrier()
        self.assertTrue(os.path.isfile(os.path.join(foo, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(foo, "fizz")))
        self.assertTrue(os.path.isfile(os.path.join(bar, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(bar, "fizz")))
        self.assertTrue(os.path.isfile(os.path.join(baz, "spam.txt")))
        self.assertTrue(os.path.isdir(os.path.join(baz, "fizz")))

    def tearDown(self):
        relentless.mpi.world.barrier()
        if relentless.mpi.world.rank_is_root:
            self._tmp.cleanup()
            del self._tmp
        self.real_f = None


if __name__ == "__main__":
    unittest.main()
