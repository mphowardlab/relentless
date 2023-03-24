from . import simulate


class EnsembleAverage(simulate.DelegatedAnalysisOperation):
    """Compute average properties.

    Parameters
    ----------
    every : int
        Interval of time steps at which to average thermodynamic properties of
        the ensemble.
    rdf : dict
        Options for computing the :class:`~relentless.model.ensemble.RDF`.
        If specified, the RDF is computed for each pair of types in the simulation.

        The dictionary **must** have the following keys:

        - ``stop``: largest bin distance
        - ``num``: number of bins

        It *may* also have the following keys:

        - ``every``: sampling frequency (defaults to the same as for properties)
    assume_constraints : bool
        If ``True``, allow the analyzer to assume that constraints implied by
        the :class:`~relentless.simulate.SimulationOperation` are valid. This
        can decrease the simulation time required. For example, the number of
        particles of each type only needs to be computed once for a
        constant-number integrator.

        .. note::

            An implementation of this operation is not *required* to do anything
            when this option is set, but it is *allowed* to.

    """

    def __init__(self, every, rdf=None, assume_constraints=False):
        self.every = every
        self.rdf = rdf
        self.assume_constraints = assume_constraints

    def _make_delegate(self, sim):
        return self._get_delegate(
            sim,
            every=self.every,
            rdf=self.rdf,
            assume_constraints=self.assume_constraints,
        )

    def _get_rdf_params(self, sim):
        if self.rdf is not None:
            # required keys
            if "num" not in self.rdf:
                raise KeyError("Number of bins is required for RDF")
            if "stop" not in self.rdf:
                raise KeyError("Stopping distance is required for RDF")

            # optional keys
            if "every" in self.rdf:
                rdf_every = self.rdf["every"]
                if rdf_every % self.every != 0:
                    raise ValueError("RDF every must be a multiple of every")
            else:
                rdf_every = self.every

            rdf_params = {
                "bins": self.rdf["num"],
                "stop": self.rdf["stop"],
                "every": rdf_every,
            }
        else:
            rdf_params = None
        return rdf_params


class Record(simulate.DelegatedAnalysisOperation):
    """Record quantities during a simulation.

    This analysis operation records a time series of values during a simulation.
    The data associated with each quantity is stored in a data key for this
    operation. Additionally, the ``"timestep"`` key gives the simulation steps
    that the quantities were recorded at.

    Parameters
    ----------
    quantities : str or list
        One or more quantities to record. Valid values are:

        - ``"potential_energy"``: Total potential energy.
        - ``"kinetic_energy"``: Total kinetic energy.
        - ``"temperature"``: Temperature.
        - ``"pressure"``: Pressure.

    every : int
        Interval of time steps at which to record values.

    """

    def __init__(self, quantities, every):
        self.quantities = quantities
        self.every = every

    def _make_delegate(self, sim):
        return self._get_delegate(
            sim,
            quantities=self.quantities,
            every=self.every,
        )

    @property
    def quantities(self):
        return self._quantities

    @quantities.setter
    def quantities(self, value):
        if isinstance(value, str):
            value = [
                value,
            ]
        else:
            value = list(value)
        self._quantities = value


class WriteTrajectory(simulate.DelegatedAnalysisOperation):
    """Write a simulation trajectory to file.

    The ``filename`` is relative to the directory where the simulation is being
    run. Regardless of the file extension, the file format is currently restricted
    to be native to the specific simulation backend. Any existing file of the
    same name will be overwritten.

    Particle positions, wrapped into the periodic simulation box, are always
    included in the trajectory. Additional properties can be opted in.

    Parameters
    ----------
    filename : str
        Name of the trajectory file to be written, as a relative path.
    every : int
        Interval of time steps at which to write a snapshot.
    velocities : bool
        Include particle velocities.
    images : bool
        Include particle images.
    types : bool
        Include particle types.
    masses : bool
        Include particle masses.

    """

    def __init__(
        self, filename, every, velocities=False, images=False, types=False, masses=False
    ):
        self.filename = filename
        self.every = every
        self.velocities = velocities
        self.images = images
        self.types = types
        self.masses = masses

    def _make_delegate(self, sim):
        return self._get_delegate(
            sim,
            filename=self.filename,
            every=self.every,
            velocities=self.velocities,
            images=self.images,
            types=self.types,
            masses=self.masses,
        )
