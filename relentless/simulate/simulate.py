__all__ = ['Simulation','SimulationInstance','SimulationOperation',
           'Potentials','PotentialTabulator','PairPotentialTabulator']

import abc

import numpy as np

class Simulation:
    """Ensemble simulation container.

    Base class that initializes and runs a simulation described by a set of
    :py:class:`SimulationOperation`s.

    Parameters
    ----------
    operations : array_like
        Array of :py:class:`SimulationOperation` to call.
    options : kwargs
        Optional arguments to attach to each instance of a simulation.

    """
    def __init__(self, operations=None, **options):
        if operations is not None:
            self.operations = operations
        else:
            self.operations = []
        self.options = options

    def run(self, ensemble, potentials, directory):
        """Run the simulation and return the result of analyze.

        A new simulation instance is created to perform the run. It is intended
        to be destroyed at the end of the run to prevent memory leaks.

        Parameters
        ----------
        ensemble : :py:class:`Ensemble`
            Simulation ensemble. Must include values for *N* and *V* even if
            these variables fluctuate.
        potentials : :py:class:`Potentials`
            The interaction potentials.
        directory : :py:class:`Directory`
            Directory to use for writing data.

        Returns
        -------
        :py:class:`Simulation`
            The simulation instance after the operations are performed.

        Raises
        ------
        TypeError
            If all operations are not :py:class:`SimulationOperation`s.

        """
        if not all([isinstance(op,SimulationOperation) for op in self.operations]):
            raise TypeError('All operations must be SimulationOperations.')

        sim = self._new_instance(ensemble, potentials, directory)
        for op in self.operations:
            op(sim)
        return sim

    @property
    def operations(self):
        """list: The operations to be performed during a simulation run."""
        return self._operations

    @operations.setter
    def operations(self, ops):
        try:
            self._operations = list(ops)
        except TypeError:
            self._operations = [ops]

    def _new_instance(self, ensemble, potentials, directory):
        return SimulationInstance(type(self),
                                  ensemble,
                                  potentials,
                                  directory,
                                  **self.options)

class SimulationInstance:
    """Specific instance of a simulation and its data.

    Parameters
    ----------
    backend : :py:class:`Simulation`
        Type of the simulation class.
    ensemble : :py:class:`Ensemble`
        Simulation ensemble. Must include values for *N* and *V* even if
        these variables fluctuate.
    potentials : :py:class:`Potentials`
        The interaction potentials.
    directory : :py:class:`Directory`
        Directory for output.
    options : kwargs
        Optional arguments for the initialize, analyze, and defined "operations" functions.

    """
    def __init__(self, backend, ensemble, potentials, directory, **options):
        self.backend = backend
        self.ensemble = ensemble
        self.potentials = potentials
        self.directory = directory
        for opt,val in options.items():
            setattr(self,opt,val)
        self._opdata = {}

    def __getitem__(self, op):
        if not op in self._opdata:
            self._opdata[op] = op.Data()
        return self._opdata[op]

class SimulationOperation(abc.ABC):
    class Data:
        pass

    @abc.abstractmethod
    def __call__(self, sim):
        pass

class Potentials:
    """Combination of multiple potentials.

    Iniitializes a :py:class:`PairPotentialTabulator` object that can store multiple potentials.
    Before the :py:class:`Potentials` object can be used, the ``rmax`` and ``num``
    attributes of all ``pair``s (that are not `None`) must be set.

    Parameters
    ----------
    pair_potentials : array_like
        The pair potentials to be combined and tabulated. (Defaults to `None`,
        resulting in an empty :py:class:`PairPotentialTabulator` object).

    """
    def __init__(self, pair_potentials=None):
        self._pair = PairPotentialTabulator(rmax=None,num=None,potentials=pair_potentials)

    @property
    def pair(self):
        """:py:class:`PairPotentialTabulator`: The combined potentials."""
        return self._pair

class PotentialTabulator:
    """Tabulates one or more potentials.

    Enables evaluation of energy, force, and derivative at different
    positional values (i.e. *x*).

    Parameters
    ----------
    start : float
        The positional value of *x* at which to begin tabulation.
    stop : float
        The positional value of *x* at which to end tabulation.
    num : int
        The number of points (value of *x*) at which to tabulate and evaluate the potential.
    potentials : :py:class:`Potential` or array_like
        The potential(s) to be tabulated. If array_like, all elements must
        be :py:class:`Potential`s. (Defaults to `None`).

    """
    def __init__(self, start, stop, num, potentials=None):
        self.start = start
        self.stop = stop
        self.num = num
        self.potentials = potentials

    @property
    def potentials(self):
        """array_like: The individual potentials that are tabulated."""
        return self._potentials

    @potentials.setter
    def potentials(self, val):
        if val is not None:
            try:
                self._potentials = list(val)
            except TypeError:
                self._potentials = [val]
        else:
            self._potentials = []

    @property
    def start(self):
        """float: The *x* value at which to start tabulation."""
        return self._start

    @start.setter
    def start(self, val):
        self._start = val
        self._compute_x = True

    @property
    def stop(self):
        """float: The *x* value at which to stop tabulation."""
        return self._stop

    @stop.setter
    def stop(self, val):
        self._stop = val
        self._compute_x = True

    @property
    def num(self):
        """int: The number of points at which to tabulate/evaluate the potential,
        must be at least 2."""
        return self._num

    @num.setter
    def num(self, val):
        if val is not None and (not isinstance(val,int) or val < 2):
            raise ValueError('Number of points must be at least 2.')
        self._num = val
        self._compute_x = True

    @property
    def x(self):
        """array_like: The values of *x* at which to evaluate energy, force, and derivative."""
        if self._compute_x:
            if self.start is None:
                raise ValueError('Start of range must be set.')
            if self.stop is None:
                raise ValueError('End of range must be set.')
            if self.num is None:
                raise ValueError('Number of points must be set.')
            self._x = np.linspace(self.start,self.stop,self.num,dtype=np.float64)
            self._compute_x = False
        return self._x

    def energy(self, key):
        """Evaluates and accumulates energy for all potentials.

        Parameters
        ----------
        key : str
            The key for which to evaluate the energy for each potential.

        Returns
        -------
        array_like
            Accumulated energy at each *x* value.

        """
        u = np.zeros_like(self.x)
        for pot in self.potentials:
            try:
                u += pot.energy(key,self.x)
            except KeyError:
                pass
        return u

    def force(self, key):
        """Evaluates and accumulates force for all potentials.

        Parameters
        ----------
        key : str
            The key for which to evaluate the force for each potential.

        Returns
        -------
        array_like
            Accumulated force at each *x* value.

        """
        f = np.zeros_like(self.x)
        for pot in self.potentials:
            try:
                f += pot.force(key,self.x)
            except KeyError:
                pass
        return f

    def derivative(self, key, var):
        """Evaluates and accumulates derivative for all potentials.

        Parameters
        ----------
        key : str
            The key for which to evaluate the derivative for each potential.

        Returns
        -------
        array_like
            Accumulated force at each *x* value.

        """
        d = np.zeros_like(self.x)
        for pot in self.potentials:
            try:
                d += pot.derivative(key,var,self.x)
            except KeyError:
                pass
        return d

class PairPotentialTabulator(PotentialTabulator):
    """Tabulates one or more pair potentials.

    Enables evaluation of energy, force, and derivative at different
    positional values (i.e. *r*).

    Parameters
    ----------
    rmax : float
        The maximum value of *r* at which to tabulate.
    num : int
        The number of points (value of *r) at which to tabulate and evaluate the potential.
    potentials : :py:class:`PairPotential` or array_like
        The pair potential(s) to be tabulated. If array_like, all elements must
        be :py:class:`PairPotential`s. (Defaults to `None`).
    fmax : float
        The maximum value of force at which to allow evaluation.

    """
    def __init__(self, rmax, num, potentials=None, fmax=None):
        super().__init__(0,rmax,num,potentials)
        self.fmax = fmax

    @property
    def r(self):
        """array_like: The values of *r* at which to evaluate energy, force, and derivative."""
        return self.x

    @property
    def rmax(self):
        """float: The maximum value of *r* at which to allow tabulation."""
        return self.stop

    @rmax.setter
    def rmax(self, val):
        self.stop = val

    @property
    def fmax(self):
        """float: The maximum value of force at which to allow evaluation."""
        return self._fmax

    @fmax.setter
    def fmax(self, val):
        if val is not None:
            self._fmax = np.fabs(val)
        else:
            self._fmax = None

    def energy(self, pair):
        """Evaluates and accumulates energy for all potentials.

        Shifts the energy to be 0 at rmax.

        Parameters
        ----------
        pair : str
            The pair for which to evaluate the energy for each potential.

        Returns
        -------
        array_like
            Accumulated energy at each *r* value.

        """
        u = super().energy(pair)
        u -= u[-1]
        return u

    def force(self, pair):
        """Evaluates and accumulates force for all potentials.

        If set, all forces are truncated to be less than or equal to ``|fmax|``.

        Parameters
        ----------
        pair : str
            The pair for which to evaluate the force for each potential.

        Returns
        -------
        array_like
            Accumulated force at each *r* value.

        """
        f = super().force(pair)
        if self.fmax is not None:
            flags = np.fabs(f) >= self.fmax
            sign = np.sign(f[flags])
            f[flags] = sign*self.fmax
        return f

    def derivative(self, pair, var):
        """Evaluates and accumulates derivative for all potentials.

        Shifts the derivative to be 0 at rmax.

        Parameters
        ----------
        pair : str
            The pair for which to evaluate the derivative for each potential.

        Returns
        -------
        array_like
            Accumulated derivative at each *r* value.

        """
        d = super().derivative(pair, var)
        d -= d[-1]
        return d
