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
    """Writes a .gsd file.

    Parameters
    ----------
    filename : str
        Name of the .gsd file to be written.
    every : int
        Interval of time steps at which to write a snapshot of the simulation.
    velocity : bool
        Log particle velocity.
    image : bool
        Log particle image.
    typeid : bool
        Log particle type ID.
    mass : bool
        Log particle mass.

    """

    def __init__(
        self, filename, every, velocity=False, image=False, typeid=False, mass=False
    ):
        super().__init__(filename, every, velocity, image, typeid, mass)
