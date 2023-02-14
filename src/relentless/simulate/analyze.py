from . import simulate


class EnsembleAverage(simulate.DelegatedAnalysisOperation):
    """Analyze the simulation ensemble.

    Parameters
    ----------
    check_thermo_every : int
        Interval of time steps at which to log thermodynamic properties of
        the simulation.
    check_rdf_every : int
        Interval of time steps at which to log the rdf of the simulation.
    rdf_dr : float
        The width (in units ``r``) of a bin in the histogram of the rdf.

    """

    def __init__(self, check_thermo_every, check_rdf_every, rdf_dr):
        self.check_thermo_every = check_thermo_every
        self.check_rdf_every = check_rdf_every
        self.rdf_dr = rdf_dr

    def _make_delegate(self, sim):
        return self._get_delegate(
            sim,
            check_thermo_every=self.check_thermo_every,
            check_rdf_every=self.check_rdf_every,
            rdf_dr=self.rdf_dr,
        )


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
