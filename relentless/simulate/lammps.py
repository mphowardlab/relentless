"""
LAMMPS
======

This module implements the :class:`LAMMPS` simulation backend and its specific
operations. It is best to interface with these operations using the frontend in
:mod:`relentless.simulate`.

.. rubric:: Developer notes

To implement your own LAMMPS operation, create an operation that derives from
:class:`LAMMPSOperation` and define the required methods.

.. autoclass:: LAMMPSOperation
    :members:
    :special-members: __call__

"""
import abc
import uuid

import lammpsio
import numpy

from relentless import mpi
from relentless.model import ensemble, extent

from . import initialize, md, simulate

try:
    import lammps

    _lammps_found = True
except ImportError:
    _lammps_found = False


class LAMMPSOperation(simulate.SimulationOperation):
    """LAMMPS simulation operation."""

    _compute_counter = 1
    _fix_counter = 1
    _group_counter = 1
    _variable_counter = 1

    def __call__(self, sim):
        """Evaluate the LAMMPS simulation operation.

        Each deriving class of :class:`LAMMPSOperation` must implement a
        :meth:`to_commands()` method that returns a list or tuple of LAMMPS
        commands that can be executed by :meth:`lammps.commands_list()`.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.SimulationInstance`
            The simulation instance.

        """
        cmds = self.to_commands(sim)
        sim.lammps.commands_list(cmds)

    @classmethod
    def new_compute_id(cls):
        """Make a unique new compute ID.

        Returns
        -------
        int
            The compute ID.

        """
        idx = int(LAMMPSOperation._compute_counter)
        LAMMPSOperation._compute_counter += 1
        return "c{}".format(idx)

    @classmethod
    def new_fix_id(cls):
        """Make a unique new fix ID.

        Returns
        -------
        int
            The fix ID.

        """
        idx = int(LAMMPSOperation._fix_counter)
        LAMMPSOperation._fix_counter += 1
        return "f{}".format(idx)

    @classmethod
    def new_group_id(cls):
        """Make a unique new fix ID.

        Returns
        -------
        int
            The fix ID.

        """
        idx = int(LAMMPSOperation._group_counter)
        LAMMPSOperation._group_counter += 1
        return "g{}".format(idx)

    @classmethod
    def new_variable_id(cls):
        """Make a unique new variable ID.

        Returns
        -------
        int
            The variable ID.

        """
        idx = int(LAMMPSOperation._variable_counter)
        LAMMPSOperation._variable_counter += 1
        return "v{}".format(idx)

    @abc.abstractmethod
    def to_commands(self, sim):
        """Create the LAMMPS commands for the simulation operation.

        All deriving classes must implement this method.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.SimulationInstance`
            The simulation instance.

        Returns
        -------
        array_like
            The LAMMPS commands for this operation.

        """
        pass


# initializers
class _Initialize(LAMMPSOperation):
    """Initialize a simulation.

    Parameters
    ----------
    lammps_types : dict
        Mapping of type names (`str`) to LAMMPS type integers.

    """

    def __init__(self, lammps_types):
        self.lammps_types = lammps_types
        self.units = "lj"
        self.atom_style = "atomic"

    def __call__(self, sim):
        super().__call__(sim)
        sim[self].lammps_types = dict(self.lammps_types)

    def to_commands(self, sim):
        cmds = [
            "units {style}".format(style=self.units),
            "boundary p p p",
            "dimension {dim}".format(dim=sim.dimension),
            "atom_style {style}".format(style=self.atom_style),
        ]

        return cmds


class InitializeFromFile(_Initialize):
    """Initialize a simulation from a LAMMPS data file.

    Because LAMMPS data files only contain the LAMMPS integer types for particles,
    you are required to specify the type map as an attribute of this operation.::

        init = InitializeFromFile('lammps.data')
        init.lammps_types = {'A': 2, 'B': 1}

    Parameters
    ----------
    filename : str
        The file from which to read the system data.

    """

    def __init__(self, filename):
        super().__init__(None)
        self.filename = filename

    def to_commands(self, sim):
        if self.lammps_types is None:
            raise ValueError("lammps_types needs to be manually specified with a file")

        cmds = super().to_commands(sim)
        cmds += ["read_data {filename}".format(filename=self.filename)]
        return cmds


class InitializeRandomly(_Initialize):
    """Initialize a randomly generated simulation box.

    If ``diameters`` is ``None``, the particles are randomly placed in the box.
    This can work pretty well for low densities, particularly if
    :class:`MinimizeEnergy` is used to remove overlaps before starting to run a
    simulation. However, it will typically fail for higher densities, where
    there are many overlaps that are hard to resolve.

    If ``diameters`` is specified for each particle type, the particles will
    be randomly packed into sites of a close-packed lattice. The insertion
    order is from big to small. No particles are allowed to overlap based on
    the diameters, which typically means the initially state will be more
    favorable than using random initialization. However, the packing procedure
    can fail if there is not enough room in the box to fit particles using
    lattice sites.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations.
    N : dict
        Number of particles of each type.
    V : :class:`~relentless.extent.Extent`
        Simulation extent.
    T : float
        Temperature. None means system is not thermalized.
    masses : dict
        Masses of each particle type. None means particles
        have unit mass.
    diameters : dict
        Diameter of each particle type. None means particles
        are randomly inserted without checking their sizes.

    """

    def __init__(self, seed, N, V, T, masses, diameters):
        super().__init__({i: idx + 1 for idx, i in enumerate(N.keys())})
        self.seed = seed
        self.N = N
        self.V = V
        self.T = T
        self.masses = masses
        self.diameters = diameters

    def to_commands(self, sim):
        if not isinstance(self.V, (extent.TriclinicBox, extent.ObliqueArea)):
            raise TypeError(
                "LAMMPS boxes must be derived from TriclinicBox or ObliqueArea"
            )
        elif (sim.dimension == 3 and not isinstance(self.V, extent.TriclinicBox)) or (
            sim.dimension == 2 and not isinstance(self.V, extent.ObliqueArea)
        ):
            raise TypeError("Mismatch between extent type and dimension")

        if mpi.world.rank_is_root:
            # make box
            if sim.dimension == 3:
                Lx, Ly, Lz, xy, xz, yz = self.V.as_array("LAMMPS")
                lo = self.V.low
            elif sim.dimension == 2:
                Lx, Ly, xy = self.V.as_array("LAMMPS")
                Lz = 1.0
                xz = 0.0
                yz = 0.0
                lo = numpy.array([self.V.low[0], self.V.low[1], -0.5 * Lz])
            else:
                raise ValueError("LAMMPS only supports 2d and 3d simulations")
            hi = lo + [Lx, Ly, Lz]
            tilt = numpy.array([xy, xz, yz])
            if not numpy.all(numpy.isclose(tilt, 0)):
                box = lammpsio.Box(lo, hi, tilt)
            else:
                box = lammpsio.Box(lo, hi)
            snap = lammpsio.Snapshot(N=sum(self.N.values()), box=box)

            # generate the positions and types
            if self.diameters is not None:
                positions, all_types = initialize.InitializeRandomly._pack_particles(
                    self.seed, self.N, self.V, self.diameters
                )
            else:
                positions, all_types = initialize.InitializeRandomly._random_particles(
                    self.seed, self.N, self.V
                )

            # set the positions
            snap.position[:, : sim.dimension] = positions

            # set the typeids
            snap.typeid = [self.lammps_types[i] for i in all_types]

            # set masses, defaulting to unit mass
            if self.masses is not None:
                snap.mass = [self.masses[i] for i in all_types]
            else:
                snap.mass[:] = 1.0

            # set velocities, defaulting to zeros
            if self.T is not None:
                rng = numpy.random.default_rng(self.seed + 1)
                # Maxwell-Boltzmann = normal with variance kT/m per component
                vel = rng.normal(
                    scale=numpy.sqrt(sim.potentials.kB * self.T),
                    size=(snap.N, sim.dimension),
                )
                vel /= numpy.sqrt(snap.mass[:, None])

                # zero the linear momentum
                v_cm = numpy.sum(snap.mass[:, None] * vel, axis=0) / numpy.sum(
                    snap.mass
                )
                vel -= v_cm
            else:
                vel = numpy.zeros((snap.N, sim.dimension))
            snap.velocity[:, : sim.dimension] = vel

            init_file = sim.directory.file(str(uuid.uuid4().hex))
            lammpsio.DataFile.create(init_file, snap, self.atom_style)
        else:
            init_file = None
        init_file = mpi.world.bcast(init_file)

        cmds = super().to_commands(sim)
        cmds += ["read_data {}".format(init_file)]
        return cmds


class MinimizeEnergy(LAMMPSOperation):
    """Perform energy minimization on a configuration.

    Valid **options** include:

    - **max_evaluations** (`int`) - the maximum number of force/energy evaluations.
      Defaults to ``100*max_iterations``.

    Parameters
    ----------
    energy_tolerance : float
        Energy convergence criterion.
    force_tolerance : float
        Force convergence criterion.
    max_iterations : int
        Maximum number of iterations to run the minimization.
    options : dict
        Additional options for energy minimzer.

    """

    def __init__(self, energy_tolerance, force_tolerance, max_iterations, options):
        self.energy_tolerance = energy_tolerance
        self.force_tolerance = force_tolerance
        self.max_iterations = max_iterations
        self.options = options
        if "max_evaluations" not in self.options:
            self.options["max_evaluations"] = None

    def to_commands(self, sim):
        if self.options["max_evaluations"] is None:
            self.options["max_evaluations"] = 100 * self.max_iterations

        cmds = [
            "minimize {etol} {ftol} {maxiter} {maxeval}".format(
                etol=self.energy_tolerance,
                ftol=self.force_tolerance,
                maxiter=self.max_iterations,
                maxeval=self.options["max_evaluations"],
            )
        ]
        if sim.dimension == 2:
            fix_2d = self.new_fix_id()
            cmds = (
                ["fix {} all enforce2d".format(fix_2d)]
                + cmds
                + ["unfix {}".format(fix_2d)]
            )

        return cmds


class _MDIntegrator(LAMMPSOperation):
    """Base LAMMPS molecular dynamics integrator.

    Parameters
    ----------
    steps : int
        Number of simulation time steps.
    timestep : float
        Simulation time step.
    analyzers : :class:`~relentless.simulate.AnalysisOperation` or list
        Analysis operations to perform with run (defaults to ``None``).

    """

    def __init__(self, steps, timestep, analyzers):
        self.steps = steps
        self.timestep = timestep
        self.analyzers = analyzers

    def __call__(self, sim):
        for analyzer in self.analyzers:
            analyzer(sim)

        super().__call__(sim)

        for analyzer in self.analyzers:
            analyzer.finalize(sim)

    @property
    def analyzers(self):
        return self._analyzers

    @analyzers.setter
    def analyzers(self, ops):
        if ops is not None:
            try:
                ops_ = list(ops)
            except TypeError:
                ops_ = [ops]
        else:
            ops_ = []

        self._analyzers = ops_

    def _run_commands(self, sim):
        cmds = ["run {N}".format(N=self.steps)]

        # wrap with fixes
        if sim.dimension == 2:
            fix_2d = self.new_fix_id()
            cmds = (
                ["fix {} all enforce2d".format(fix_2d)]
                + cmds
                + ["unfix {}".format(fix_2d)]
            )

        return cmds


class RunLangevinDynamics(_MDIntegrator):
    """Perform a Langevin dynamics simulation.

    See :class:`relentless.simulate.RunLangevinDynamics` for details.

    Parameters
    ----------
    steps : int
        Number of simulation time steps.
    timestep : float
        Simulation time step.
    T : float
        Temperature.
    friction : float or dict
        Sets drag coefficient for each particle type (shared or per-type).
    seed : int
        Seed used to randomly generate a uniform force.
    analyzers : :class:`~relentless.simulate.AnalysisOperation` or list
        Analysis operations to perform with run (defaults to ``None``).

    """

    def __init__(self, steps, timestep, T, friction, seed, analyzers):
        super().__init__(steps, timestep, analyzers)
        self.T = T
        self.friction = friction
        self.seed = seed

    def to_commands(self, sim):
        # obtain per-type mass (arrays 1-indexed using lammps convention)
        Ntypes = len(sim.types)
        mass = sim.lammps.numpy.extract_atom("mass")
        if mass is None or mass.shape != (Ntypes + 1,):
            raise ValueError("Per-type masses not set.")
        mass = numpy.squeeze(mass)

        # obtain per-type friction factor
        friction = numpy.zeros_like(mass)
        for t in sim.types:
            try:
                friction[sim[sim.initializer].lammps_types[t]] = self.friction[t]
            except TypeError:
                friction[sim[sim.initializer].lammps_types[t]] = self.friction
            except KeyError:
                raise KeyError("The friction factor for type {} is not set.".format(t))

        # compute per-type damping parameter and rescale if multiple types
        damp = numpy.divide(mass, friction, where=(friction > 0))
        damp_ref = damp[1]
        if Ntypes > 1:
            scale = damp / damp_ref
            scale_str = " ".join(
                ["scale {} {}".format(i + 1, s) for i, s in enumerate(scale[2:])]
            )
        else:
            scale_str = ""

        fix_ids = {"nve": self.new_fix_id(), "langevin": self.new_fix_id()}
        cmds = [
            "timestep {}".format(self.timestep),
            "fix {idx} {group_idx} nve".format(idx=fix_ids["nve"], group_idx="all"),
            (
                "fix {idx} {group_idx} langevin {t_start} {t_stop}"
                " {damp} {seed} {scaling}"
            ).format(
                idx=fix_ids["langevin"],
                group_idx="all",
                t_start=self.T,
                t_stop=self.T,
                damp=damp_ref,
                seed=self.seed,
                scaling=scale_str,
            ),
        ]
        cmds += self._run_commands(sim)
        cmds += ["unfix {}".format(idx) for idx in fix_ids.values()]
        return cmds


class RunMolecularDynamics(_MDIntegrator):
    """Perform a molecular dynamics simulation.

    This method supports:

    - NVE integration
    - NVT integration with Nosé-Hoover or Berendsen thermostat
    - NPH integration with MTK or Berendsen barostat
    - NPT integration with Nosé-Hoover or Berendsen thermostat and
      MTK or Berendsen barostat

    Parameters
    ----------
    steps : int
        Number of simulation time steps.
    timestep : float
        Simulation time step.
    thermostat : :class:`~relentless.simulate.Thermostat`
        Thermostat for temperature control. None means no thermostat.
    barostat : :class:`~relentless.simulate.Barostat`
        Barostat for pressure control. None means no barostat.
    analyzers : :class:`~relentless.simulate.AnalysisOperation` or list
        Analysis operations to perform with run (defaults to ``None``).

    Raises
    ------
    TypeError
        If an appropriate combination of thermostat and barostat is not set.

    """

    def __init__(self, steps, timestep, thermostat, barostat, analyzers):
        super().__init__(steps, timestep, analyzers)
        self.thermostat = thermostat
        self.barostat = barostat

    def to_commands(self, sim):
        fix_ids = {"ig": self.new_fix_id()}

        cmds = ["timestep {}".format(self.timestep)]
        if (
            self.thermostat is None
            or isinstance(self.thermostat, md.BerendsenThermostat)
        ) and (
            self.barostat is None or isinstance(self.barostat, md.BerendsenBarostat)
        ):
            cmds += [
                "fix {idx} {group_idx} nve".format(idx=fix_ids["ig"], group_idx="all")
            ]
        elif isinstance(self.thermostat, md.NoseHooverThermostat) and (
            self.barostat is None or isinstance(self.barostat, md.BerendsenBarostat)
        ):
            cmds += [
                "fix {idx} {group_idx} nvt temp {Tstart} {Tstop} {Tdamp}".format(
                    idx=fix_ids["ig"],
                    group_idx="all",
                    Tstart=self.thermostat.T,
                    Tstop=self.thermostat.T,
                    Tdamp=self.thermostat.tau,
                )
            ]
        elif (
            self.thermostat is None
            or isinstance(self.thermostat, md.BerendsenThermostat)
        ) and isinstance(self.barostat, md.MTKBarostat):
            cmds += [
                "fix {idx} {group_idx} nph iso {Pstart} {Pstop} {Pdamp}".format(
                    idx=fix_ids["ig"],
                    group_idx="all",
                    Pstart=self.barostat.P,
                    Pstop=self.barostat.P,
                    Pdamp=self.barostat.tau,
                )
            ]
        elif isinstance(self.thermostat, md.NoseHooverThermostat) and isinstance(
            self.barostat, md.MTKBarostat
        ):
            cmds += [
                (
                    "fix {idx} {group_idx} npt temp {Tstart} {Tstop} {Tdamp}"
                    " iso {Pstart} {Pstop} {Pdamp}"
                ).format(
                    idx=fix_ids["ig"],
                    group_idx="all",
                    Tstart=self.thermostat.T,
                    Tstop=self.thermostat.T,
                    Tdamp=self.thermostat.tau,
                    Pstart=self.barostat.P,
                    Pstop=self.barostat.P,
                    Pdamp=self.barostat.tau,
                )
            ]
        else:
            raise TypeError(
                "An appropriate combination of thermostat and barostat must be set."
            )

        if isinstance(self.thermostat, md.BerendsenThermostat):
            fix_ids["berendsen_temp"] = self.new_fix_id()
            cmds += [
                "fix {idx} {group_idx} temp/berendsen {Tstart} {Tstop} {Tdamp}".format(
                    idx=fix_ids["berendsen_temp"],
                    group_idx="all",
                    Tstart=self.thermostat.T,
                    Tstop=self.thermostat.T,
                    Tdamp=self.thermostat.tau,
                )
            ]
        if isinstance(self.barostat, md.BerendsenBarostat):
            fix_ids["berendsen_press"] = self.new_fix_id()
            cmds += [
                (
                    "fix {idx} {group_idx} press/berendsen iso"
                    " {Pstart} {Pstop} {Pdamp}"
                ).format(
                    idx=fix_ids["berendsen_press"],
                    group_idx="all",
                    Pstart=self.barostat.P,
                    Pstop=self.barostat.P,
                    Pdamp=self.barostat.tau,
                )
            ]

        cmds += self._run_commands(sim)
        cmds += ["unfix {}".format(idx) for idx in fix_ids.values()]
        return cmds


class EnsembleAverage(simulate.AnalysisOperation, LAMMPSOperation):
    """Analyzes the simulation ensemble and rdf at specified timestep intervals.

    Parameters
    ----------
    check_thermo_every : int
        Interval of time steps at which to log thermodynamic properties of the
        simulation.
    check_rdf_every : int
        Interval of time steps at which to log the rdf of the simulation.
    rdf_dr : float
        The width (in units ``r``) of a bin in the histogram of the rdf.

    Raises
    ------
    RuntimeError
        If more than one LAMMPS :class:`AddEnsembleAnalyzer` is initialized
        at the same time.

    """

    def __init__(self, check_thermo_every, check_rdf_every, rdf_dr):
        self.check_thermo_every = check_thermo_every
        self.check_rdf_every = check_rdf_every
        self.rdf_dr = rdf_dr

    def to_commands(self, sim):
        fix_ids = {"thermo_avg": self.new_fix_id(), "rdf_avg": self.new_fix_id()}
        compute_ids = {"rdf": self.new_compute_id()}
        var_ids = {
            "T": self.new_variable_id(),
            "P": self.new_variable_id(),
            "Lx": self.new_variable_id(),
            "Ly": self.new_variable_id(),
            "Lz": self.new_variable_id(),
            "xy": self.new_variable_id(),
            "xz": self.new_variable_id(),
            "yz": self.new_variable_id(),
        }

        group_ids = {}
        for i in sim.types:
            typeid = sim[sim.initializer].lammps_types[i]
            typekey = "N_{}".format(typeid)
            group_ids[typekey] = self.new_group_id()
            var_ids[typekey] = self.new_variable_id()

        # generate temporary file names
        if mpi.world.rank_is_root:
            file_ = {
                "thermo": sim.directory.file(str(uuid.uuid4().hex)),
                "rdf": sim.directory.file(str(uuid.uuid4().hex)),
            }
        else:
            file_ = None
        file_ = mpi.world.bcast(file_)

        # thermodynamic properties
        sim[self].thermo_file = file_["thermo"]
        cmds = [
            "variable {} equal temp".format(var_ids["T"]),
            "variable {} equal press".format(var_ids["P"]),
            "variable {} equal lx".format(var_ids["Lx"]),
            "variable {} equal ly".format(var_ids["Ly"]),
            "variable {} equal lz".format(var_ids["Lz"]),
            "variable {} equal xy".format(var_ids["xy"]),
            "variable {} equal xz".format(var_ids["xz"]),
            "variable {} equal yz".format(var_ids["yz"]),
        ]
        N_vars = []
        for i in sim.types:
            typeid = sim[sim.initializer].lammps_types[i]
            typekey = "N_{}".format(typeid)
            cmds += [
                "group {gid} type {typeid}".format(
                    gid=group_ids[typekey], typeid=typeid
                ),
                'variable {vid} equal "count({gid})"'.format(
                    vid=var_ids[typekey], gid=group_ids[typekey]
                ),
            ]
            N_vars.append("v_{" + typekey + "}")
        cmds += [
            (
                "fix {fixid} all ave/time {every} 1 {every}"
                " v_{T} v_{P} v_{Lx} v_{Ly} v_{Lz} v_{xy} v_{xz} v_{yz}"
                + " "
                + " ".join(N_vars)
                + " mode scalar ave running"
                ' file {filename} overwrite format " %.16e"'
            ).format(
                fixid=fix_ids["thermo_avg"],
                every=self.check_thermo_every,
                filename=sim[self].thermo_file,
                **var_ids,
            )
        ]

        # pair distribution function
        rmax = sim.potentials.pair.r[-1]
        sim[self].num_bins = numpy.round(rmax / self.rdf_dr).astype(int)
        sim[self].rdf_file = file_["rdf"]
        sim[self].rdf_pairs = tuple(sim.pairs)
        # string format lammps arguments based on pairs
        # _pairs is the list of all pairs by LAMMPS type id, in ensemble order
        # _computes is the RDF values for each pair, with the r bin centers prepended
        _pairs = []
        _computes = ["c_{}[1]".format(compute_ids["rdf"])]
        for idx, (i, j) in enumerate(sim[self].rdf_pairs):
            _pairs.append(
                "{} {}".format(
                    sim[sim.initializer].lammps_types[i],
                    sim[sim.initializer].lammps_types[j],
                )
            )
            _computes.append("c_{}[{}]".format(compute_ids["rdf"], 2 * (idx + 1)))
        cmds += [
            "compute {rdf} all rdf {bins} {pairs}".format(
                rdf=compute_ids["rdf"], bins=sim[self].num_bins, pairs=" ".join(_pairs)
            ),
            (
                "fix {fixid} all ave/time {every} 1 {every}"
                " {computes} mode vector ave running off 1"
                ' file {filename} overwrite format " %.16e"'
            ).format(
                fixid=fix_ids["rdf_avg"],
                every=self.check_rdf_every,
                computes=" ".join(_computes),
                filename=sim[self].rdf_file,
            ),
        ]

        return cmds

    def finalize(self, sim):
        # extract thermo properties
        # we skip the first 2 rows, which are LAMMPS junk, and slice out the
        # timestep from col. 0
        thermo = mpi.world.loadtxt(sim[self].thermo_file, skiprows=2)[1:]
        N = {i: Ni for i, Ni in zip(sim.types, thermo[8 : 8 + len(sim.types)])}
        V = extent.TriclinicBox(
            Lx=thermo[2],
            Ly=thermo[3],
            Lz=thermo[4],
            xy=thermo[5],
            xz=thermo[6],
            yz=thermo[7],
            convention="LAMMPS",
        )
        ens = ensemble.Ensemble(N=N, T=thermo[0], P=thermo[1], V=V)

        # extract rdfs
        # LAMMPS injects a column for the row index, so we start at column 1 for r
        # we skip the first 4 rows, which are LAMMPS junk, and slice out the
        # first column
        rdf = mpi.world.loadtxt(sim[self].rdf_file, skiprows=4)[:, 1:]
        for i, pair in enumerate(sim[self].rdf_pairs):
            ens.rdf[pair] = ensemble.RDF(rdf[:, 0], rdf[:, i + 1])

        sim[self].ensemble = ens


class LAMMPS(simulate.Simulation):
    """Simulation using LAMMPS.

    A simulation is performed using `LAMMPS <https://docs.lammps.org>`_.
    LAMMPS is a molecular dynamics program that can execute on both CPUs and
    GPUs, as a single process or with MPI parallelism. The launch configuration
    will be automatically selected for you when the simulation is run.

    LAMMPS must be built with its
    `Python interface <https://docs.lammps.org/Python_head.html>`_ and must be
    version 29 Sep 2021 or newer.

    .. warning::

        LAMMPS requires that tabulated pair potentials do not include an entry for
        :math:`r = 0`. Make sure to set
        :attr:`~relentless.simulate.PairPotentialTabulator.rmin` to a small value
        larger than 0.

    Parameters
    ----------
    initializer : :class:`~relentless.simulate.SimulationOperation`
        Operation that initializes the simulation.
    operations : array_like
        :class:`~relentless.simulate.SimulationOperation` to execute for run.
        Defaults to ``None``, which means nothing is done after initialization.
    dimension : int
        Dimensionality of the simulation. Defaults to 3.
    quiet : bool
        If ``True``, silence LAMMPS screen output. Setting this to ``False`` can
        be helpful for debugging but would be very noisy in a long production
        simulation.

    Raises
    ------
    ImportError
        If the :mod:`lammps` package is not found.

    """

    def __init__(self, initializer, operations=None, dimension=3, quiet=True):
        if not _lammps_found:
            raise ImportError("LAMMPS not found.")

        super().__init__(initializer, operations)
        self.dimension = dimension
        self.quiet = quiet

    def _new_instance(self, initializer, potentials, directory):
        sim = super()._new_instance(initializer, potentials, directory)

        # add the lammps engine to the instance
        if self.quiet:
            # create lammps instance with all output disabled
            launch_args = [
                "-echo",
                "none",
                "-log",
                "none",
                "-screen",
                "none",
                "-nocite",
            ]
        else:
            launch_args = [
                "-echo",
                "screen",
                "-log",
                sim.directory.file("log.lammps"),
                "-nocite",
            ]
        sim.lammps = lammps.lammps(cmdargs=launch_args)
        if sim.lammps.version() < 20210929:
            raise ImportError("Only LAMMPS 29 Sep 2021 or newer is supported.")

        # run the initializer
        sim.dimension = self.dimension
        initializer(sim)
        sim.types = sim[initializer].lammps_types.keys()

        # attach the potentials
        self._attach_potentials(sim)

        return sim

    def _attach_potentials(self, sim):
        """Adds tabulated pair potentials to the simulation object.

        Parameters
        ----------
        sim: :class:`~relentless.simulate.SimulationInstance`
            The simulation instance.

        Raises
        ------
        ValueError
            If there are not at least two points in the tabulated potential.
        ValueError
            If the pair potentials do not have equally spaced ``r``.

        """
        if sim.potentials.pair.rmin == 0:
            raise ValueError("LAMMPS requires rmin > 0 for pair potentials")
        r = sim.potentials.pair.r
        Nr = len(r)
        if Nr == 1:
            raise ValueError(
                "LAMMPS requires at least two points in the tabulated potential."
            )

        # check that all r are equally spaced
        dr = r[1:] - r[:-1]
        if not numpy.all(numpy.isclose(dr, dr[0])):
            raise ValueError("LAMMPS requires equally spaced r in pair potentials.")

        def pair_map(sim, pair):
            # Map lammps type indexes as a pair, lowest type first
            i, j = pair
            id_i = sim[sim.initializer].lammps_types[i]
            id_j = sim[sim.initializer].lammps_types[j]
            if id_i > id_j:
                id_i, id_j = id_j, id_i

            return id_i, id_j

        # write all potentials into a file
        if mpi.world.rank_is_root:
            file_ = sim.directory.file(str(uuid.uuid4().hex))
            with open(file_, "w") as fw:
                fw.write("# LAMMPS tabulated pair potentials\n")
                rcut = {}
                for i, j in sim.pairs:
                    id_i, id_j = pair_map(sim, (i, j))
                    fw.write(
                        ("# pair ({i},{j})\n" "\n" "TABLE_{id_i}_{id_j}\n").format(
                            i=i, j=j, id_i=id_i, id_j=id_j
                        )
                    )
                    fw.write(
                        "N {N} R {rmin} {rmax}\n\n".format(N=Nr, rmin=r[0], rmax=r[-1])
                    )

                    u = sim.potentials.pair.energy((i, j))
                    f = sim.potentials.pair.force((i, j))
                    for idx in range(Nr):
                        fw.write(
                            "{idx} {r} {u} {f}\n".format(
                                idx=idx + 1, r=r[idx], u=u[idx], f=f[idx]
                            )
                        )

                    # find r where potential and force are zero
                    nonzero_r = numpy.flatnonzero(
                        numpy.logical_and(~numpy.isclose(u, 0), ~numpy.isclose(f, 0))
                    )
                    if len(nonzero_r) > 1:
                        # cutoff at last nonzero r (cannot be first r)
                        rcut[(i, j)] = r[nonzero_r[-1]]
                    else:
                        # if first or second r is nonzero, cutoff at second
                        rcut[(i, j)] = r[1]
        else:
            rcut = None
            file_ = None
        rcut = mpi.world.bcast(rcut)
        file_ = mpi.world.bcast(file_)

        # process all lammps commands
        cmds = [
            "neighbor {skin} multi".format(skin=sim.potentials.pair.neighbor_buffer)
        ]
        cmds += ["pair_style table linear {N}".format(N=Nr)]

        for i, j in sim.pairs:
            # get lammps type indexes, lowest type first
            id_i, id_j = pair_map(sim, (i, j))
            cmds += [
                (
                    "pair_coeff {id_i} {id_j} {filename}"
                    " TABLE_{id_i}_{id_j} {cutoff}"
                ).format(id_i=id_i, id_j=id_j, filename=file_, cutoff=rcut[(i, j)])
            ]

        sim.lammps.commands_list(cmds)

    # initialize
    InitializeFromFile = InitializeFromFile
    InitializeRandomly = InitializeRandomly

    # md
    MinimizeEnergy = MinimizeEnergy
    RunLangevinDynamics = RunLangevinDynamics
    RunMolecularDynamics = RunMolecularDynamics

    # analyze
    EnsembleAverage = EnsembleAverage
