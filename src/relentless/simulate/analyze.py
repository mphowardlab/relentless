from . import simulate


class EnsembleAverage(simulate.GenericAnalysisOperation):
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
        super().__init__(check_thermo_every, check_rdf_every, rdf_dr)


class WriteTrajectory(simulate.GenericAnalysisOperation):
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
        super().__init__(filename, every, velocities, images, types, masses)
