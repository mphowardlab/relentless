"""
Data management
===============
The :class:`Directory` class provides an interface for creating hierarchical
filesystem directories and files within those directories using either an absolute
or relative path.

.. autosummary::
    :nosignatures:

    Directory

.. autoclass:: Directory
    :members:

"""
import os
import shutil

from . import mpi

class Directory:
    """Context for a filesystem directory.

    The directory specified by ``path`` (which can be either absolute or relative)
    is created if it does not already exist. This process is recursive, so
    ``path`` may include multiple directories that do not yet exist. This object
    represents the final directory in ``path``.

    A :class:`Directory` is a context that can be used to manage the current
    working directory. Entering the context changes the current working
    directory to ``path``, and exiting restores the working directory before the
    context was entered.

    Parameters
    ----------
    path : str
        Absolute or relative directory path.

    Raises
    ------
    OSError
        If the specified path is not a valid directory.

    Examples
    --------
    Creating a directory::

        d = Directory('foo')

    Using the context to open a file ``foo/bar.txt`` in a directory::

        with Directory('foo') as d:
            f = open('bar.txt')

    """
    def __init__(self, path):
        self._start = []

        # ensure path exists at time directory is created (synchronizing)
        path = os.path.realpath(path)
        if mpi.world.rank_is_root:
            if not os.path.exists(path):
                os.makedirs(path)
            dir_error = not os.path.isdir(path)
        else:
            dir_error = None
        mpi.world.bcast(dir_error)
        if dir_error:
            raise OSError('The specified path is not a valid directory')
        self._path = path

    @classmethod
    def cast(cls, directory):
        """Try to cast an object to a directory.

        Ensure that a `str` or :class:`Directory` is a :class:`Directory`. No
        action is taken if the object is already a :class:`Directory`. Otherwise,
        a new one is constructed.

        Parameters
        ----------
        directory : str or :class:`Directory`
            Object to ensure is a directory

        Returns
        -------
        :class:`Directory`
            The cast object.

        """
        if not isinstance(directory, Directory):
            directory = Directory(directory)
        return directory

    def __enter__(self):
        """Enter the directory context.

        The working directory is changed to the ``path`` of this object.

        Returns
        -------
        :class:`Directory`
            This directory.

        """
        self._start.append(os.getcwd())
        os.chdir(self.path)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Exit the directory context.

        If possible, the working directory is reset to the path before entering
        the context. The change is silently ignored if the original directory
        no longer exists.

        """
        try:
            os.chdir(self._start.pop())
        except OSError:
            pass

    def _in_context(self):
        """bool: True if object is being used as a context."""
        return len(self._start) > 0

    @property
    def path(self):
        """str: Real path to the directory."""
        return self._path

    def file(self, name):
        """Get the absolute path to a file in the directory.

        This method is convenient for abstracting references to a file in the
        directory.

        Parameters
        ----------
        name : str
            Name of the file.

        Returns
        -------
        str
            The absolute path to the file ``name``.

        Examples
        --------
        Opening a file by absolute path::

            d = Directory('foo')
            f = open(d.file('bar.txt'))

        """
        return os.path.join(self.path, name)

    def directory(self, name):
        """Get a child directory.

        This method is convenient for abstracting references to child
        directories.

        Parameters
        ----------
        name : str
            Name of the directory.

        Returns
        -------
        :class:`Directory`
            A new directory relative to this one.

        Examples
        --------
        Making nested directories ``foo/bar``::

            foo = Directory('foo')
            bar = foo.directory('bar')

        """
        return Directory(os.path.join(self.path, name))

    def clear_contents(self):
        r"""Clear the contents of a directory.

        This method **removes** all the contents of a directory (files and
        directories), so it should be used carefully!

        """
        # delete on root rank and wait
        if mpi.world.rank_is_root:
            for entry in os.scandir(self.path):
                if entry.is_file():
                    os.remove(entry.path)
                elif entry.is_dir():
                    shutil.rmtree(entry.path)
        mpi.world.barrier()

    def move_contents(self, dest):
        """Move the contents of the directory.

        Parameters
        ----------
        dest : :class:`Directory` or :class:`str`
            Destination directory.

        """
        dest = Directory.cast(dest)
        # move on root rank and wait
        if mpi.world.rank_is_root:
            for entry in os.scandir(self.path):
                shutil.move(entry.path, dest.path)
        mpi.world.barrier()

    def copy_contents(self, dest):
        """Copy the contents of the directory.

        Parameters
        ----------
        dest : :class:`Directory` or :class:`str`
            Destination directory.

        """
        dest = Directory.cast(dest)
        # copy using root rank and wait
        if mpi.world.rank_is_root:
            for entry in os.scandir(self.path):
                if entry.is_file():
                    shutil.copy2(entry.path, dest.path)
                elif entry.is_dir():
                    shutil.copytree(entry.path, os.path.join(dest.path,entry.name))
        mpi.world.barrier()
