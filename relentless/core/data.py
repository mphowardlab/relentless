"""Core data management.

Todo
----
Improve this documentation for developers!

"""
__all__ = ['Directory','Project']

import os
import shutil

class Directory:
    """Context for a filesystem directory.

    The directory specified by `path` (which can be either absolute or relative)
    is created if it does not already exist. This process is recursive, so
    `path` may include multiple directories that do not yet exist. This object
    represents the final directory in `path`.

    A :py:class:`Directory` is a context that can be used to manage the current
    working directory. Entering the context changes the current working
    directory to *path*, and exiting restores the working directory before the
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

    Using the context to open a file `foo/bar.txt` in a directory::

        with Directory('foo') as d:
            f = open('bar.txt')

    """
    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self._start = []

        if not os.path.isdir(self.path):
            raise OSError('The specified path is not a valid directory.')

    def __enter__(self):
        """Enter the directory context.

        The working directory is changed to the `path` of this object.

        Returns
        -------
        :py:class:`Directory`
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
            if self._start[-1] is not os.getcwd():
                os.chdir(self._start.pop())
        except OSError:
            pass

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
            The absolute path to the file `name`.

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
        :py:class:`Directory`
            A new directory relative to this one.

        Examples
        --------
        Making nested directories `foo/bar`::

            foo = Directory('foo')
            bar = foo.directory('bar')

        """
        return Directory(os.path.join(self.path, name))

    def clear(self):
        r"""Clear the contents of a directory.

        This method **removes** all the contents of a directory (files and
        directories), so it should be used carefully!

        """
        with self:
            for entry in os.scandir(self.path):
                if entry.is_file():
                    os.remove(entry.name)
                elif entry.is_dir():
                    shutil.rmtree(entry.name)

class Project:
    r"""Project data.

    The project data will be saved primarily in two directories. The first is
    ``workspace``, which is for results that should be saved. This data includes
    things like parameter values at a given iteration or the error in the optimization.
    The second is ``scratch``, which is meant to be used for temporary data such
    as simulation trajectories or program output that does not need to be preserved.

    The names of these spaces are meant to suggest the filesystem structure of
    many high-performance computers (e.g., GPFS, Lustre), where the ``work``
    disks are usually shared and persistent while the ``scratch`` disks are
    possibly node-local and temporary (or scrubbed periodically). ``scratch``
    disks are usually faster for I/O intensive processes, so it is recommended
    to specify this directory if you have access to it. Otherwise, a
    pseudo-``scratch`` directory will be created in the ``workspace``, but it
    will not be any more performant than the ``workspace``.

    The :py:class:`Project` does not specify a data schema. The schema must
    be created by consumers of the :py:class:`Project`. A :py:class:`Project`
    also does not guarantee that the ``scratch`` space will actually be
    cleared. It is up to the user (or the system) to remove this data.

    Parameters
    ----------
    workspace : str or :py:class:`Directory`
        Directory for persistent project data.
        Defaults to ``./workspace`` (using UNIX filesystem notation).
    scratch : str or :py:class:`Directory`
        Directory for temporary (scratch) project data.
        Defaults to ``./workspace/scratch`` (using UNIX filesystem notation).

    """
    def __init__(self, workspace=None, scratch=None):
        # use specified workspace, otherwise use the current directory
        if workspace is not None:
            if isinstance(workspace, Directory):
                self._work = workspace
            else:
                self._work = Directory(workspace)
        else:
            self._work = Directory(os.path.join(os.getcwd(),'workspace'))

        # use system scratch if specified, otherwise make one
        if scratch is not None:
            if isinstance(scratch, Directory):
                self._scratch = scratch
            else:
                self._scratch = Directory(scratch)
        else:
            self._scratch = self.workspace.directory('scratch')

    @property
    def workspace(self):
        r""":py:class:`Directory` Work space."""
        return self._work

    @property
    def scratch(self):
        r""":py:class:`Directory` Scratch space."""
        return self._scratch
