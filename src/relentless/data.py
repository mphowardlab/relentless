"""
===================================
Data management (`relentless.data`)
===================================

.. currentmodule:: relentless.data

.. autosummary::
    :toctree: generated/

    Directory

"""

import os
import pathlib
import shutil
import uuid


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
    create : bool
        If True, ensure the directory is created on the filesystem.

    Raises
    ------
    OSError
        If the specified path is not a valid directory.

    Examples
    --------
    Creating a directory::

        d = Directory('foo')

    Create a directory on the root rank and wait to proceed::

        d = Directory('bar', create=relentless.mpi.world.rank_is_root)
        relentless.mpi.world.barrier()

    Using the context to open a file ``foo/bar.txt`` in a directory::

        with Directory('foo') as d:
            f = open('bar.txt')

    """

    def __init__(self, path, create=True):
        self._start = []
        self._path = os.path.realpath(path)
        if create:
            os.makedirs(self._path, exist_ok=True)

    @classmethod
    def cast(cls, directory, create=True):
        """Try to cast an object to a directory.

        Ensure that a `str` or :class:`Directory` is a :class:`Directory`. No
        action is taken if the object is already a :class:`Directory`. Otherwise,
        a new one is constructed.

        Parameters
        ----------
        directory : str or :class:`Directory`
            Object to ensure is a directory
        create : bool
            If True, ensure the directory is created on the filesystem.

        Returns
        -------
        :class:`Directory`
            The cast object.

        """
        if not isinstance(directory, Directory):
            directory = Directory(directory)
        if create:
            os.makedirs(directory.path, exist_ok=True)
        return directory

    def __enter__(self):
        """Enter the directory context.

        The working directory is changed to the ``path`` of this object.

        Returns
        -------
        :class:`Directory`
            This directory.

        """
        if not os.path.isdir(self.path):
            raise OSError("Directory does not exist")
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

    def temporary_file(self, suffix=None):
        """Get a temporary filename in the directory.

        This method generates a random filename using :func:`uuid.uuid4`. Unlike
        a Python temporary file, this file is not automatically created or
        destroyed. That responsibility is delegated to the caller.

        Parameters
        ----------
        suffix : str
            If specified, the suffix to add to the filename. If ``None`` (default),
            no suffix is added.

        Returns
        -------
        str
            The absolute path to the temporary file.

        """
        filename = str(uuid.uuid4().hex)
        if suffix is not None:
            filename += suffix
        return self.file(filename)

    def directory(self, name, create=True):
        """Get a child directory.

        This method is convenient for abstracting references to child
        directories.

        Parameters
        ----------
        name : str
            Name of the directory.
        create : bool
            If True, ensure the directory is created on the filesystem.

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
        return Directory(os.path.join(self.path, name), create=create)

    def clear_contents(self):
        r"""Clear the contents of a directory.

        This method **removes** all the contents of a directory (files and
        directories), so it should be used carefully!

        """
        # delete on root rank and wait
        for entry in os.scandir(self.path):
            if entry.is_file():
                os.remove(entry.path)
            elif entry.is_dir():
                shutil.rmtree(entry.path)

    def move_contents(self, dest):
        """Move the contents of the directory.

        Parameters
        ----------
        dest : :class:`Directory` or :class:`str`
            Destination directory.

        Raises
        ------
        OSError
            If the destination exists and does not match the type of the source.

        """
        dest = Directory.cast(dest, create=True)
        for entry in os.scandir(self.path):
            dest_entry = pathlib.Path(dest.path, entry.name)
            if dest_entry.exists():
                if dest_entry.is_dir() and entry.is_dir():
                    shutil.copytree(entry.path, dest_entry, dirs_exist_ok=True)
                    shutil.rmtree(entry.path)
                elif dest_entry.is_file() and entry.is_file():
                    shutil.move(entry.path, dest_entry)
                else:
                    raise OSError(
                        "Destination "
                        "{} exists and does not match type of source {}".format(
                            dest_entry, entry.name
                        )
                    )
            else:
                shutil.move(entry.path, dest_entry)

    def copy_contents(self, dest):
        """Copy the contents of the directory.

        Parameters
        ----------
        dest : :class:`Directory` or :class:`str`
            Destination directory.

        """
        dest = Directory.cast(dest, create=True)
        for entry in os.scandir(self.path):
            if entry.is_file():
                shutil.copy2(entry.path, dest.path)
            elif entry.is_dir():
                shutil.copytree(entry.path, os.path.join(dest.path, entry.name))

    def is_empty(self):
        """Check if the directory is empty.

        Returns
        -------
        :bool
            True if the directory is empty.

        """
        return len(os.listdir(self.path)) != 0
