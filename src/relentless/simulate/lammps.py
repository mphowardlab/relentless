import abc
import datetime
import subprocess
import uuid

import numpy

from relentless import collections, mpi
from relentless.model import ensemble, extent

from . import initialize, md, simulate

try:
    import lammps

    _lammps_found = True
except ImportError:
    _lammps_found = False

try:
    import lammpsio

    _lammpsio_found = True
except ImportError:
    _lammpsio_found = False


# initializers
class InitializationOperation(simulate.InitializationOperation):
    """Initialize a simulation."""

    def __call__(self, sim):
        SimulationOperation.__call__(self, sim)

    def _call_commands(self, sim):
        cmds = [
            "units {style}".format(style=sim["engine"]["units"]),
            "boundary p p p",
            "dimension {dim}".format(dim=sim["engine"]["dimension"]),
            "atom_style {style}".format(style=sim["engine"]["atom_style"]),
        ]

        sim.dimension = sim["engine"]["dimension"]
        cmds += self.initialize_commands(sim)
        sim.types = sim["engine"]["types"].keys()

        sim.masses = collections.FixedKeyDict(sim.types)
        # file is opened only valid on root and result is broadcast
        masses = {}
        if mpi.world.rank_is_root:
            snap = lammpsio.DataFile(sim[self]["_datafile"]).read()
            for i in sim.types:
                mi = snap.mass[snap.typeid == sim["engine"]["types"][i]]
                if len(mi) == 0:
                    raise KeyError("Type {} not present in simulation".format(i))
                elif not numpy.all(mi == mi[0]):
                    raise ValueError("All masses for a type must be equal")
                masses[i] = mi[0]
        masses = mpi.world.bcast(masses)
        sim.masses.update(masses)

        # attach the potentials
        if sim.potentials.pair.start == 0:
            raise ValueError("LAMMPS requires start > 0 for pair potentials")
        r, u, f = sim.potentials.pair.pairwise_energy_and_force(
            sim.types, x=sim.potentials.pair.squared_space, tight=True, minimum_num=2
        )
        Nr = len(r)
        sim[self]["_potentials_rmax"] = r[-1]

        def pair_map(sim, pair):
            # Map lammps type indexes as a pair, lowest type first
            i, j = pair
            id_i = sim["engine"]["types"][i]
            id_j = sim["engine"]["types"][j]
            if id_i > id_j:
                id_i, id_j = id_j, id_i

            return id_i, id_j

        # write all potentials into a file
        if mpi.world.rank_is_root:
            file_ = sim.directory.file(str(uuid.uuid4().hex))
            with open(file_, "w") as fw:
                fw.write("# LAMMPS tabulated pair potentials\n")
                for i, j in sim.pairs:
                    if numpy.any(numpy.isinf(u[i, j])) or numpy.any(
                        numpy.isinf(f[i, j])
                    ):
                        raise ValueError(
                            "Pair potential/force is infinite at evaluated r"
                        )

                    id_i, id_j = pair_map(sim, (i, j))
                    fw.write(
                        ("# pair ({i},{j})\n" "\n" "TABLE_{id_i}_{id_j}\n").format(
                            i=i, j=j, id_i=id_i, id_j=id_j
                        )
                    )
                    fw.write(
                        "N {N} RSQ {rmin} {rmax}\n\n".format(
                            N=Nr,
                            rmin=r[0],
                            rmax=r[-1],
                        )
                    )

                    for idx, (r_, u_, f_) in enumerate(
                        zip(r, u[i, j], f[i, j]), start=1
                    ):
                        fw.write(
                            "{idx} {r} {u} {f}\n".format(idx=idx, r=r_, u=u_, f=f_)
                        )
        else:
            file_ = None
        file_ = mpi.world.bcast(file_)

        # process all lammps commands
        cmds += [
            "neighbor {skin} multi".format(skin=sim.potentials.pair.neighbor_buffer),
            "neigh_modify delay 0 every 1 check yes",
        ]
        cmds += ["pair_style table linear {N}".format(N=Nr)]

        for i, j in sim.pairs:
            # get lammps type indexes, lowest type first
            id_i, id_j = pair_map(sim, (i, j))
            cmds += [
                ("pair_coeff {id_i} {id_j} {filename}" " TABLE_{id_i}_{id_j}").format(
                    id_i=id_i, id_j=id_j, filename=file_
                )
            ]

        return cmds

    @abc.abstractmethod
    def initialize_commands(self, sim):
        pass


class InitializeFromFile(InitializationOperation):
    """Initialize a simulation from a LAMMPS data file.

    Because LAMMPS data files only contain the LAMMPS integer types for particles,
    you are required to specify the type map as an attribute of this operation.::

        init = InitializeFromFile('lammps.data')
        init.types = {'A': 2, 'B': 1}

    Parameters
    ----------
    filename : str
        The file from which to read the system data.

    """

    def __init__(self, filename):
        self.filename = filename

    def initialize_commands(self, sim):
        if sim["engine"]["types"] is None:
            raise ValueError("types needs to be manually specified with a file")
        sim[self]["_datafile"] = self.filename
        return ["read_data {filename}".format(filename=self.filename)]


class InitializeRandomly(InitializationOperation):
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
        self.seed = seed
        self.N = N
        self.V = V
        self.T = T
        self.masses = masses
        self.diameters = diameters

    def initialize_commands(self, sim):
        if not isinstance(self.V, (extent.TriclinicBox, extent.ObliqueArea)):
            raise TypeError(
                "LAMMPS boxes must be derived from TriclinicBox or ObliqueArea"
            )
        elif (sim.dimension == 3 and not isinstance(self.V, extent.TriclinicBox)) or (
            sim.dimension == 2 and not isinstance(self.V, extent.ObliqueArea)
        ):
            raise TypeError("Mismatch between extent type and dimension")

        if sim["engine"]["types"] is None:
            sim["engine"]["types"] = {i: idx + 1 for idx, i in enumerate(self.N.keys())}

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
            snap.typeid = [sim["engine"]["types"][i] for i in all_types]

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
            lammpsio.DataFile.create(init_file, snap, sim["engine"]["atom_style"])
        else:
            init_file = None
        init_file = mpi.world.bcast(init_file)

        sim[self]["_datafile"] = init_file
        return ["read_data {}".format(init_file)]


# simulation operations
class SimulationOperation(simulate.SimulationOperation):
    """LAMMPS simulation operation."""

    _compute_counter = 1
    _dump_counter = 1
    _fix_counter = 1
    _group_counter = 1
    _variable_counter = 1

    def __call__(self, sim):
        """Evaluate the LAMMPS simulation operation.

        Each deriving class of :class:`SimulationOperation` must implement a
        :meth:`_call_commands()` method that returns a list or tuple of LAMMPS
        commands that can be executed by :meth:`lammps.commands_list()`.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.SimulationInstance`
            The simulation instance.

        """
        cmds = self._call_commands(sim)
        if cmds is None or len(cmds) == 0:
            return

        sim["engine"]["_lammps_commands"] += cmds
        if sim["engine"]["use_python"]:
            sim["engine"]["_lammps"].commands_list(cmds)

    @classmethod
    def new_compute_id(cls):
        """Make a unique new compute ID.

        Returns
        -------
        int
            The compute ID.

        """
        idx = int(SimulationOperation._compute_counter)
        SimulationOperation._compute_counter += 1
        return "c{}".format(idx)

    @classmethod
    def new_dump_id(cls):
        """Make a unique new dump ID.

        Returns
        -------
        int
            The dump ID.

        """
        idx = int(SimulationOperation._dump_counter)
        SimulationOperation._dump_counter += 1
        return "d{}".format(idx)

    @classmethod
    def new_fix_id(cls):
        """Make a unique new fix ID.

        Returns
        -------
        int
            The fix ID.

        """
        idx = int(SimulationOperation._fix_counter)
        SimulationOperation._fix_counter += 1
        return "f{}".format(idx)

    @classmethod
    def new_group_id(cls):
        """Make a unique new fix ID.

        Returns
        -------
        int
            The fix ID.

        """
        idx = int(SimulationOperation._group_counter)
        SimulationOperation._group_counter += 1
        return "g{}".format(idx)

    @classmethod
    def new_variable_id(cls):
        """Make a unique new variable ID.

        Returns
        -------
        int
            The variable ID.

        """
        idx = int(SimulationOperation._variable_counter)
        SimulationOperation._variable_counter += 1
        return "v{}".format(idx)

    @abc.abstractmethod
    def _call_commands(self, sim):
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


class MinimizeEnergy(SimulationOperation):
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

    def _call_commands(self, sim):
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


class _Integrator(SimulationOperation):
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
        super().__init__(analyzers)
        self.steps = steps
        self.timestep = timestep

    def __call__(self, sim):
        for analyzer in self.analyzers:
            analyzer.pre_run(sim, self)

        super().__call__(sim)

        for analyzer in self.analyzers:
            analyzer.post_run(sim, self)

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

    @staticmethod
    def _make_T(thermostat):
        """Cast thermostat into a T parameter for LAMMPS integrators."""
        # force the type of thermostat, in case it's a float
        if not isinstance(thermostat, md.Thermostat):
            thermostat = md.Thermostat(thermostat)

        if thermostat.anneal:
            return (thermostat.T[0], thermostat.T[1])
        else:
            return (thermostat.T, thermostat.T)


class RunBrownianDynamics(_Integrator):
    """Perform a Brownian dynamics simulation.

    See :class:`relentless.simulate.RunBrownianDynamics` for details.

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

    def _call_commands(self, sim):
        if "BROWNIAN" not in sim["engine"]["packages"]:
            raise NotImplementedError("LAMMPS BROWNIAN package is not installed.")
        elif sim["engine"]["version"] < 20220623:
            raise NotImplementedError(
                "LAMMPS versions prior to 23Jun2022 stable release do not"
                " properly support Brownian dynamics."
            )

        T = self._make_T(self.T)
        if T[0] != T[1]:
            raise NotImplementedError(
                "Brownian dynamics cannot do temperature annealing in LAMMPS."
            )

        try:
            len(self.friction)
            same_friction = False
        except TypeError:
            same_friction = True

        fix_ids = []
        group_ids = []
        cmd_template = "fix {fixid} {groupid} brownian {T} {seed} gamma_t {friction}"

        cmds = ["timestep {}".format(self.timestep)]
        if same_friction:
            fixid = self.new_fix_id()
            cmds.append(
                cmd_template.format(
                    fixid=fixid,
                    groupid="all",
                    T=T[0],
                    seed=self.seed,
                    friction=self.friction,
                )
            )
            fix_ids.append(fixid)
        else:
            for i, t in enumerate(sim.types):
                typeidx = sim["engine"]["types"][t]
                groupid = self.new_group_id()
                fixid = self.new_fix_id()
                cmds += [
                    f"group {groupid} type {typeidx}",
                    cmd_template.format(
                        fixid=fixid,
                        groupid=groupid,
                        T=T[0],
                        seed=self.seed + i,
                        friction=self.friction[t],
                    ),
                ]
                fix_ids.append(fixid)
                group_ids.append(groupid)
        cmds += self._run_commands(sim)
        cmds += ["unfix {}".format(idx) for idx in fix_ids]
        cmds += ["group {} delete".format(idx) for idx in group_ids]
        return cmds


class RunLangevinDynamics(_Integrator):
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

    def _call_commands(self, sim):
        # obtain per-type friction factor
        Ntypes = len(sim.types)
        mass = numpy.zeros(Ntypes)
        friction = numpy.zeros(Ntypes)
        for t in sim.types:
            typeidx = sim["engine"]["types"][t] - 1
            try:
                friction[typeidx] = self.friction[t]
            except TypeError:
                friction[typeidx] = self.friction
            except KeyError:
                raise KeyError("The friction factor for type {} is not set.".format(t))

            mass[typeidx] = sim.masses[t]

        # compute per-type damping parameter and rescale if multiple types
        damp = numpy.divide(mass, friction, where=(friction > 0))
        damp_ref = damp[0]
        if Ntypes > 1:
            scale = damp / damp_ref
            scale_str = " ".join(
                ["scale {} {}".format(i + 2, s) for i, s in enumerate(scale[1:])]
            )
        else:
            scale_str = ""

        T = self._make_T(self.T)
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
                t_start=T[0],
                t_stop=T[1],
                damp=damp_ref,
                seed=self.seed,
                scaling=scale_str,
            ),
        ]
        cmds += self._run_commands(sim)
        cmds += ["unfix {}".format(idx) for idx in fix_ids.values()]
        return cmds


class RunMolecularDynamics(_Integrator):
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

    def _call_commands(self, sim):
        fix_ids = {"ig": self.new_fix_id()}

        if self.thermostat is not None:
            T = self._make_T(self.thermostat)
        else:
            T = None

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
                    Tstart=T[0],
                    Tstop=T[1],
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
                    Tstart=T[0],
                    Tstop=T[1],
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
                    Tstart=T[0],
                    Tstop=T[1],
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


# analyzers
class AnalysisOperation(simulate.AnalysisOperation):
    def pre_run(self, sim, sim_op):
        cmds = self._pre_run_commands(sim, sim_op)
        if cmds is None or len(cmds) == 0:
            return

        sim["engine"]["_lammps_commands"] += cmds
        if sim["engine"]["use_python"]:
            sim["engine"]["_lammps"].commands_list(cmds)

    def post_run(self, sim, sim_op):
        cmds = self._post_run_commands(sim, sim_op)
        if cmds is None or len(cmds) == 0:
            return

        sim["engine"]["_lammps_commands"] += cmds
        if sim["engine"]["use_python"]:
            sim["engine"]["_lammps"].commands_list(cmds)

    @abc.abstractmethod
    def _pre_run_commands(self, sim, sim_op):
        pass

    @abc.abstractmethod
    def _post_run_commands(self, sim, sim_op):
        pass


class EnsembleAverage(AnalysisOperation):
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

    def _pre_run_commands(self, sim, sim_op):
        fix_ids = {
            "thermo_avg": SimulationOperation.new_fix_id(),
            "rdf_avg": SimulationOperation.new_fix_id(),
        }
        compute_ids = {"rdf": SimulationOperation.new_compute_id()}
        var_ids = {
            "T": SimulationOperation.new_variable_id(),
            "P": SimulationOperation.new_variable_id(),
            "Lx": SimulationOperation.new_variable_id(),
            "Ly": SimulationOperation.new_variable_id(),
            "Lz": SimulationOperation.new_variable_id(),
            "xy": SimulationOperation.new_variable_id(),
            "xz": SimulationOperation.new_variable_id(),
            "yz": SimulationOperation.new_variable_id(),
        }

        group_ids = {}
        for i in sim.types:
            typeid = sim["engine"]["types"][i]
            typekey = "N_{}".format(typeid)
            group_ids[typekey] = SimulationOperation.new_group_id()
            var_ids[typekey] = SimulationOperation.new_variable_id()

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
        sim[self]["_thermo_file"] = file_["thermo"]
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
            typeid = sim["engine"]["types"][i]
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
                filename=sim[self]["_thermo_file"],
                **var_ids,
            )
        ]

        # pair distribution function
        rmax = sim[sim.initializer]["_potentials_rmax"]
        num_bins = numpy.round(rmax / self.rdf_dr).astype(int)
        if num_bins == 1:
            raise ValueError("Only 1 RDF bin, increase the discretization")
        sim[self]["_rdf_file"] = file_["rdf"]
        sim[self]["_rdf_pairs"] = tuple(sim.pairs)
        # string format lammps arguments based on pairs
        # _pairs is the list of all pairs by LAMMPS type id, in ensemble order
        # _computes is the RDF values for each pair, with the r bin centers prepended
        _pairs = []
        _computes = ["c_{}[1]".format(compute_ids["rdf"])]
        for idx, (i, j) in enumerate(sim[self]["_rdf_pairs"]):
            _pairs.append(
                "{} {}".format(
                    sim["engine"]["types"][i],
                    sim["engine"]["types"][j],
                )
            )
            _computes.append("c_{}[{}]".format(compute_ids["rdf"], 2 * (idx + 1)))
        cmds += [
            "compute {rdf} all rdf {bins} {pairs}".format(
                rdf=compute_ids["rdf"], bins=num_bins, pairs=" ".join(_pairs)
            ),
            (
                "fix {fixid} all ave/time {every} 1 {every}"
                " {computes} mode vector ave running off 1"
                ' file {filename} overwrite format " %.16e"'
            ).format(
                fixid=fix_ids["rdf_avg"],
                every=self.check_rdf_every,
                computes=" ".join(_computes),
                filename=sim[self]["_rdf_file"],
            ),
        ]

        # save ids so we can remove them later
        sim[self]["_fix_ids"] = fix_ids
        sim[self]["_compute_ids"] = compute_ids
        sim[self]["_var_ids"] = var_ids
        sim[self]["_group_ids"] = group_ids

        return cmds

    def _post_run_commands(self, sim, sim_op):
        cmds = []
        # unfix
        for fixid in sim[self]["_fix_ids"].values():
            cmds.append("unfix {}".format(fixid))
        del sim[self]["_fix_ids"]
        # uncommand
        for compute_id in sim[self]["_compute_ids"].values():
            cmds.append("uncompute {}".format(compute_id))
        del sim[self]["_compute_ids"]
        # delete variables
        for var_id in sim[self]["_var_ids"].values():
            cmds.append("variable {} delete".format(var_id))
        del sim[self]["_var_ids"]
        # delete groups
        for group_id in sim[self]["_group_ids"].values():
            cmds.append("group {} delete".format(group_id))
        del sim[self]["_group_ids"]

        return cmds

    def process(self, sim, sim_op):
        # extract thermo properties
        # we skip the first 2 rows, which are LAMMPS junk, and slice out the
        # timestep from col. 0
        try:
            thermo = mpi.world.loadtxt(sim[self]["_thermo_file"], skiprows=2)[1:]
        except Exception as e:
            raise RuntimeError("No LAMMPS thermo file generated") from e
        N = {i: Ni for i, Ni in zip(sim.types, thermo[8 : 8 + len(sim.types)])}
        if sim.dimension == 3:
            V = extent.TriclinicBox(
                Lx=thermo[2],
                Ly=thermo[3],
                Lz=thermo[4],
                xy=thermo[5],
                xz=thermo[6],
                yz=thermo[7],
                convention="LAMMPS",
            )
        else:
            V = extent.ObliqueArea(
                Lx=thermo[2],
                Ly=thermo[3],
                xy=thermo[5],
                convention="LAMMPS",
            )
        ens = ensemble.Ensemble(N=N, T=thermo[0], P=thermo[1], V=V)

        # extract rdfs
        # LAMMPS injects a column for the row index, so we start at column 1 for r
        # we skip the first 4 rows, which are LAMMPS junk, and slice out the
        # first column
        try:
            rdf = mpi.world.loadtxt(sim[self]["_rdf_file"], skiprows=4)[:, 1:]
        except Exception as e:
            raise RuntimeError("LAMMPS RDF file could not be read") from e
        for i, pair in enumerate(sim[self]["_rdf_pairs"]):
            ens.rdf[pair] = ensemble.RDF(rdf[:, 0], rdf[:, i + 1])

        sim[self]["ensemble"] = ens
        sim[self]["num_thermo_samples"] = None
        sim[self]["num_rdf_samples"] = None


class Record(AnalysisOperation):
    def __init__(self, quantities, every):
        self.quantities = quantities
        self.every = every

    def _pre_run_commands(self, sim, sim_op):
        # translate quantities into lammps variables
        quantity_map = {
            "potential_energy": "pe",
            "kinetic_energy": "ke",
            "temperature": "temp",
            "pressure": "press",
        }
        var_ids = {}
        cmds = []
        for q in self.quantities:
            var_ids[q] = SimulationOperation.new_variable_id()
            cmds.append("variable {} equal {}".format(var_ids[q], quantity_map[q]))

        # write quantities to file with fix ave/time
        if mpi.world.rank == 0:
            file_ = sim.directory.file(str(uuid.uuid4().hex))
        else:
            file_ = None
        sim[self]["_fix_id"] = SimulationOperation.new_fix_id()
        sim[self]["_log_file"] = mpi.world.bcast(file_)
        cmds.append(
            (
                "fix {fixid} all ave/time {every} 1 {every} {vars}"
                ' mode scalar ave one file {filename} format " %.18e"'
            ).format(
                fixid=sim[self]["_fix_id"],
                every=self.every,
                filename=sim[self]["_log_file"],
                vars=" ".join(["v_" + str(var_ids[q]) for q in self.quantities]),
            )
        )

        # stash variable ids to delete them later
        sim[self]["_var_ids"] = var_ids
        return cmds

    def _post_run_commands(self, sim, sim_op):
        cmds = ["unfix {}".format(sim[self]["_fix_id"])]
        del sim[self]["_fix_id"]
        # delete variables
        for var_id in sim[self]["_var_ids"].values():
            cmds.append("variable {} delete".format(var_id))
        del sim[self]["_var_ids"]

        return cmds

    def process(self, sim, sim_op):
        data = mpi.world.loadtxt(sim[self]["_log_file"])
        sim[self]["timestep"] = data[:, 0].astype(int)
        for i, q in enumerate(self.quantities, start=1):
            sim[self][q] = data[:, i]


class WriteTrajectory(AnalysisOperation):
    """Writes a LAMMPS dump file.

    When all options are set to True the file has the following format::

        ITEM: ATOMS id type mass x y z vx vy vz ix iy iz

    where ``id`` is the atom ID, ``x y z`` are positions, ``vx vy vz`` are
    velocities, and ``ix iy iz`` are images.

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

    def __init__(self, filename, every, velocities, images, types, masses):
        self.filename = filename
        self.every = every
        self.velocities = velocities
        self.images = images
        self.types = types
        self.masses = masses

    def _pre_run_commands(self, sim, sim_op):
        dump_format = "id"
        if self.types is True:
            dump_format += " type"
        if self.masses is True:
            dump_format += " mass"
        # position is always dynamic
        dump_format += " x y z"
        if self.velocities is True:
            dump_format += " vx vy vz"
        if self.images is True:
            dump_format += " ix iy iz"

        dump_id = SimulationOperation.new_dump_id()
        cmds = [
            "dump {} all custom {} {} {}".format(
                dump_id, self.every, sim.directory.file(self.filename), dump_format
            ),
            "dump_modify {} append no pbc yes flush yes".format(dump_id),
        ]

        sim[self]["_dump_id"] = dump_id

        return cmds

    def _post_run_commands(self, sim, sim_op):
        cmds = ["undump {}".format(sim[self]["_dump_id"])]
        del sim[self]["_dump_id"]
        return cmds

    def process(self, sim, sim_op):
        pass


class LAMMPS(simulate.Simulation):
    """Simulation using LAMMPS.

    A simulation is performed using `LAMMPS <https://docs.lammps.org>`_.
    LAMMPS is a molecular dynamics program that can execute on both CPUs and
    GPUs, as a single process or with MPI parallelism. The launch configuration
    will be automatically selected for you when the simulation is run.

    The version of LAMMPS must be 29 Sep 2021 or newer. It is recommended to build
    LAMMPS with its `Python interface <https://docs.lammps.org/Python_head.html>`_.
    However, it is possible to run LAMMPS as a binary by specifying ``executable``::

        relentless.simulate.LAMMPS(init, ops, executable="lmp_serial")

    This can be helpful if you do not have a build of LAMMPS with Python support
    enabled; however, it will typically be a bit slower than running LAMMPS via
    Python. To run LAMMPS as an executable with MPI support, you should **not**
    launch ``relentless`` with ``mpirexec``, and instead should include the
    ``mpiexec` command and options in the ``executable``::

        relentless.simulate.LAMMPS(init, ops, executable="mpiexec -n 8 lmp_mpi")


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
    types : dict
        Mapping from relentless types to LAMMPS integer types. This mapping may
        be used during initialization (and is required for some operations).
    executable : str
        LAMMPS executable. If specified, LAMMPS will be run as a binary
        application rather than its Python library.

    Raises
    ------
    ImportError
        If the :mod:`lammps` package is not found.

    """

    def __init__(
        self,
        initializer,
        operations=None,
        dimension=3,
        quiet=True,
        types=None,
        executable=None,
    ):
        # test executable if it is specified
        if executable is not None:
            # disallow launching LAMMPS under MPI. this may be slightly too strict,
            # but we can relax it later if needed.
            if mpi.world.enabled:
                raise RuntimeError(
                    "LAMMPS cannot be run under MPI with an executable."
                    " Put mpiexec in the executable instead."
                )
            # test the executable
            result = subprocess.run(
                executable + " -help", shell=True, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise OSError(
                    "LAMMPS executable {} failed to launch.".format(executable)
                )
            self.executable = executable
            lines = result.stdout.splitlines()
            for i, line in enumerate(result.stdout.splitlines()):
                if i == 1:
                    # these split indexes are hardcoded based on standard help output
                    version_str = line.split("-")[2].strip()
                    # then this coerces the version into the LAMMPS integer format
                    self.version = int(
                        datetime.datetime.strptime(version_str, "%d %b %Y").strftime(
                            "%Y%m%d"
                        )
                    )
                elif line == "Installed packages:":
                    installed_packages = []
                    for line_ in lines[i + 2 :]:
                        if len(line_) == 0:
                            break
                        installed_packages += line_.strip().split()
                    self.packages = tuple(installed_packages)
        else:
            if not _lammps_found:
                raise ImportError("LAMMPS not found.")

            self.executable = None
            lmp = lammps.lammps(
                cmdargs=[
                    "-echo",
                    "none",
                    "-log",
                    "none",
                    "-screen",
                    "none",
                    "-nocite",
                ]
            )
            self.version = lmp.version()
            self.packages = tuple(lmp.installed_packages)
            del lmp
        if self.version < 20210929:
            raise ImportError("Only LAMMPS 29 Sep 2021 or newer is supported.")

        if not _lammpsio_found:
            raise ImportError("lammpsio not found.")

        super().__init__(initializer, operations)
        self.dimension = dimension
        self.quiet = quiet
        self.types = types

    def _post_run(self, sim):
        # force all the lammps commands to execute, since the operations did
        # not actually do it
        if not sim["engine"]["use_python"]:
            # send the commands to file
            if mpi.world.rank_is_root:
                file_ = sim.directory.file(str(uuid.uuid4().hex))
                with open(file_, "w") as f:
                    for cmd in sim["engine"]["_lammps_commands"]:
                        f.write(cmd + "\n")
            else:
                file_ = None
            file_ = mpi.world.bcast(file_)

            # then run lammps as an executable
            run_cmd = sim["engine"]["_lammps"] + ["-i", file_]
            subprocess.run(" ".join(run_cmd), shell=True, check=True)

        # then keep going
        super()._post_run(sim)

    def _initialize_engine(self, sim):
        if self.quiet:
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

        sim["engine"]["version"] = self.version
        sim["engine"]["packages"] = self.packages
        if self.executable is not None:
            sim["engine"]["use_python"] = False
            sim["engine"]["_lammps"] = [self.executable] + launch_args
        else:
            sim["engine"]["use_python"] = True
            sim["engine"]["_lammps"] = lammps.lammps(cmdargs=launch_args)
        sim["engine"]["_lammps_commands"] = []

        sim["engine"]["types"] = self.types
        sim["engine"]["units"] = "lj"
        sim["engine"]["atom_style"] = "atomic"
        sim["engine"]["dimension"] = self.dimension

    # initialize
    _InitializeFromFile = InitializeFromFile
    _InitializeRandomly = InitializeRandomly

    # md
    _MinimizeEnergy = MinimizeEnergy
    _RunBrownianDynamics = RunBrownianDynamics
    _RunLangevinDynamics = RunLangevinDynamics
    _RunMolecularDynamics = RunMolecularDynamics

    # analyze
    _EnsembleAverage = EnsembleAverage
    _Record = Record
    _WriteTrajectory = WriteTrajectory
