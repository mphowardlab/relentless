"""
Data management
===============
The :class:`Directory` class provides an interface for creating hierarchical
filesystem directories and files within those directories using either an absolute
or relative path. Additionally, project simulation and optimization data
can be stored using a :class:`Project`, which includes options for ``workspace``
and ``scratch`` directories.

.. autosummary::
    :nosignatures:

    Directory
    Project

.. autoclass:: Directory
    :members:

.. autoclass:: Project
    :members:

"""
import os
import shutil

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
        self.path = path

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

    @path.setter
    def path(self, value):
        if self._in_context():
            raise OSError('Cannot set path while using as context')

        _path = os.path.realpath(value)
        if not os.path.exists(_path):
            os.makedirs(_path)
        if not os.path.isdir(_path):
            raise OSError('The specified path is not a valid directory.')

        self._path = _path

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
        for entry in os.scandir(self.path):
            if entry.is_file():
                os.remove(entry.path)
            elif entry.is_dir():
                shutil.rmtree(entry.path)

    def move_contents(self, dest):
        """Move the contents of the directory.

        Parameters
        ----------
        dest : :class:`Directory` or :type`str`
            Destination directory.

        """
        if not isinstance(dest, Directory):
            dest = Directory(dest)

        for entry in os.scandir(self.path):
            shutil.move(entry.path, dest.path)

    def copy_contents(self, dest):
        """Copy the contents of the directory.

        Parameters
        ----------
        dest : :class:`Directory` or :type:`str`
            Destination directory.

        """
        if not isinstance(dest, Directory):
            dest = Directory(dest)

        for entry in os.scandir(self.path):
            if entry.is_file():
                shutil.copy2(entry.path, dest.path)
            elif entry.is_dir():
                shutil.copytree(entry.path, os.path.join(dest.path,entry.name))

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

    The :class:`Project` does not specify a data schema. The schema must
    be created by consumers of the :class:`Project`. A :class:`Project`
    also does not guarantee that the ``scratch`` space will actually be
    cleared. It is up to the user (or the system) to remove this data.

    Parameters
    ----------
    workspace : str or :class:`Directory`
        Directory for persistent project data.
        Defaults to ``./workspace`` (using UNIX filesystem notation).
    scratch : str or :class:`Directory`
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
        r""":class:`Directory` Work space."""
        return self._work

    @property
    def scratch(self):
        r""":class:`Directory` Scratch space."""
        return self._scratch
